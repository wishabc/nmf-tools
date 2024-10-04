import numpy as np

from nmf_tools.plotting.modular_plot import DataLoader, DataBundle
from genome_tools.genomic_interval import df_to_genomic_intervals, filter_df_to_interval, df_to_variant_intervals

from genome_tools.data.extractors import tabix_extractor as TabixExtractor
from genome_tools.utils.signal import smooth_and_aggregate_per_nucleotide_signal

from footprint_tools.cli.post import posterior_stats as PosteriorStats
from footprint_tools.stats import posterior

from nmf_tools.plotting.extract_reads import extract_allelic_reads

class IdeogramLoader(DataLoader):
    __required_fields__ = ['ideogram_data']


class GencodeLoader(DataLoader):
    __required_fields__ = ['gencode_annotation_file']


class FinemapLoader(DataLoader):
    __required_fields__ = ['finemap_df']

    def _load(self, data: DataBundle, region, trait, cs_id):
        finemap_df = self.preprocessor.finemap_df.query(
            f'region == "{region}" & trait == "{trait}" & cs_id == {cs_id}'
        ).drop_duplicates('end')
        data.unique_finemap_df = finemap_df
        return data
    

class SignalLoader(DataLoader):
    __required_fields__ = []
    
    def _load(self, data, signal_files, smooth=True, step=20, bandwidth=150):
        if not smooth:
            bandwidth = 1
        segs = smooth_and_aggregate_per_nucleotide_signal(data.interval, signal_files,
                                                          step=step, bandwidth=bandwidth)
        data.signal = segs
        return data


class ComponentTracksLoader(DataLoader):
    __required_fields__ = ['cutcounts_files']

    def _load(self, data: DataBundle, smooth=True, step=20, bandwidth=150, nmf_components=None):
        if not smooth:
            bandwidth = 1

        cutcounts_files = self.preprocessor.cutcounts_files
        if nmf_components is None:
            nmf_components = cutcounts_files.keys()

        data.nmf_components = nmf_components

        data.component_tracks = []
        for component in nmf_components:
            segs = smooth_and_aggregate_per_nucleotide_signal(data.interval,
                                                              cutcounts_files[component],
                                                              step=step, bandwidth=bandwidth)
            data.component_tracks.append(segs)
        return data


class SegmentsLoader(DataLoader):
    __required_fields__ = [] # needed to be set in subclasses, first element will be used as the df to extract intervals from
    __intervals_attr__ = 'intervals'

    def _load(self, data: DataBundle, extra_columns=None, rectprops_columns=None):
        if rectprops_columns is None:
            rectprops_columns = []
        if extra_columns is None:
            extra_columns = []
        segments_df = getattr(self.preprocessor, self.__required_fields__[0])
        setattr(data, self.__intervals_attr__, 
                df_to_genomic_intervals(
            segments_df.reset_index(drop=True).reset_index(),
            self.interval,
            extra_columns=['index'] + extra_columns + rectprops_columns
        ))

        if rectprops_columns:
            for interval in getattr(data, self.__intervals_attr__):
                interval.rectprops = {col: getattr(interval, col) for col in rectprops_columns}
        return data


class DHSIndexLoader(SegmentsLoader):
    __required_fields__ = ['annotated_dhs_index']
    __intervals_attr__ = 'dhs_intervals'


class FootprintsLoader(SegmentsLoader):
    __required_fields__ = ['full_footprints_index']
    __intervals_attr__ = 'footprint_intervals'


class DHSLoadingsLoader(DataLoader):
    __required_fields__ = ['H']


# TODO: sample data file actually is not needed, it should be any pandas df file whatsoever
class FootprintDatasetLoader(DataLoader):
    __required_fields__ = ['fp_sample_data_file', 'fp_sample_data']

    def _load(self, data: DataBundle, fp_samples, fdr_cutoff=0.05):
        dl = PosteriorStats(
            self.preprocessor.fp_sample_data_file,
            self.preprocessor.fp_sample_data.loc[fp_samples],
            fdr_cutoff=fdr_cutoff,
        )
        dl._open_tabix_files()
        obs, exp, fdr, w = dl._load_data(data.interval)

        prior = posterior.compute_prior_weighted(fdr, w, cutoff=0.05) #????
        delta = posterior.compute_delta_prior(
            obs, exp, fdr, dl.betas, cutoff=0.1 #????
        )
        
        ll_on = posterior.log_likelihood(obs, exp, dl.disp_models, delta=delta, w=3)
        ll_off = posterior.log_likelihood(obs, exp, dl.disp_models, w=3)

        post = -posterior.posterior(prior, ll_on, ll_off)
        post[post <= 0] = 0.0

        z = 1 - np.exp(-post)

        data.pp = z
        data.obs = obs
        data.exp = exp

        return data


class MotifLoader(DataLoader):
    __required_fields__ = ['motif_annotations_path', 'motif_meta']

    def _load(self, data: DataBundle):
        interval_motif_annotations = TabixExtractor(self.preprocessor.motif_annotations_path,
                                                    columns=[
                                                        'chrom', 'start', 'end', 'fp_id',
                                                        'motif_chr', 'motif_start', 'motif_end',
                                                        'pfm', 'dg', 'orient', 'sequence'
                                                        ])[self.interval]
        interval_motif_annotations['dg'] = interval_motif_annotations['dg'].astype(float)
        interval_motif_annotations['start'] = interval_motif_annotations['start'].astype(int)
        interval_motif_annotations['end'] = interval_motif_annotations['end'].astype(int)
        interval_motif_annotations['motif_start'] = interval_motif_annotations['motif_start'].astype(int)
        interval_motif_annotations['motif_end'] = interval_motif_annotations['motif_end'].astype(int)
        interval_motif_annotations = interval_motif_annotations.groupby('fp_id', group_keys=False).apply(lambda x: x.nlargest(1, 'dg'))
        interval_motif_annotations['motif_id'] = interval_motif_annotations['pfm'].str.replace('.pfm', '')
        interval_motif_annotations = interval_motif_annotations.merge(self.preprocessor.motif_meta, left_on='motif_id', right_index=True)
        data.motif_intervals = df_to_genomic_intervals(interval_motif_annotations, self.interval, extra_columns=['orient', 'motif_start', 'motif_end', 'tf_name', 'pwm'])
        return data


class AggregatedCAVLoader(DataLoader):
    __required_fields__ = ['cavs_data']

    def _load(self, data: DataBundle, fdr_tr=0.1, color='k', notsignif_color='#C0C0C0'):
        filtered_cavs = filter_df_to_interval(self.preprocessor.cavs_data, self.interval)
        filtered_cavs['is_significant'] = filtered_cavs['min_fdr'] <= fdr_tr
        filtered_cavs['sig_es'] = np.clip(np.where(filtered_cavs['is_significant'], np.abs(filtered_cavs['logit_es_combined']), 0), 0, 2)
        group_ids_df = filtered_cavs.query('is_significant').groupby(['#chr', 'start', 'end', 'ref', 'alt'])['group_id'].apply(lambda x: ','.join(map(str, x))).reset_index()
        filtered_cavs = filtered_cavs.groupby(['#chr', 'start', 'end', 'ref', 'alt'], group_keys=False).apply(lambda x: x.nlargest(1, 'sig_es'))
        filtered_cavs = filtered_cavs.merge(group_ids_df, on=['#chr', 'start', 'end', 'ref', 'alt'], how='left', suffixes=('', '_list'))
        
        filtered_cavs['value'] = np.abs(filtered_cavs['logit_es_combined'])
        filtered_cavs['color'] = np.where(filtered_cavs['is_significant'], color, notsignif_color)
        data.cavs_intervals = df_to_variant_intervals(filtered_cavs, extra_columns=['value', 'color'])
        return data
    

class PerSampleCAVLoader(DataLoader):
    __required_fields__ = ['nonaggregated_cavs_data']

    def _load(self, data: DataBundle, sample_id, fdr_tr=0.1, color='k', notsignif_color='#C0C0C0'):
        filtered_cavs = TabixExtractor(self.preprocessor.nonaggregated_cavs_data)[self.interval].query(f'sample_id == "{sample_id}"')
        filtered_cavs['is_significant'] = filtered_cavs['FDR_sample'] <= fdr_tr
        filtered_cavs['sig_es'] = np.clip(np.where(filtered_cavs['is_significant'], np.abs(filtered_cavs['logit_es']), 0), 0, 2)
        
        filtered_cavs['value'] = np.abs(filtered_cavs['logit_es'])
        filtered_cavs['color'] = np.where(filtered_cavs['is_significant'], color, notsignif_color)
        data.cavs_intervals = df_to_variant_intervals(filtered_cavs, extra_columns=['value', 'color'])
        return data


class AllelicReadsLoader(DataLoader):
    __required_fields__ = ['samples_metadata']

    def _load(self, data: DataBundle, sample_ids, variant_interval):
        if isinstance(sample_ids, (str, int, float)):
            sample_ids = [sample_ids]
        cram_paths = self.preprocessor.samples_metadata.loc[sample_ids, 'cram_file']
        reads = {}
        for sample_id, cram_path in zip(sample_ids, cram_paths):
            reads[sample_id] = extract_allelic_reads(cram_path, variant_interval, data.interval)
        data.reads = reads
        return data
