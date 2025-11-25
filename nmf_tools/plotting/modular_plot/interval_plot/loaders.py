import numpy as np
import pandas as pd

from nmf_tools.plotting.modular_plot import PlotDataLoader, DataBundle
from genome_tools.genomic_interval import df_to_genomic_intervals, filter_df_to_interval, df_to_variant_intervals

from genome_tools.data.extractors import TabixExtractor, FastaExtractor, VariantGenotypeExtractor

from genome_tools.utils.signal import smooth_and_aggregate_per_nucleotide_signal


from nmf_tools.plotting.modular_plot.interval_plot.basic_loaders import SignalLoader, SegmentsLoader


class FinemapLoader(PlotDataLoader):

    def _load(self, data: DataBundle, finemap_df: pd.DataFrame, region, trait, cs_id):
        finemap_df = finemap_df.query(
            f'region == "{region}" & trait == "{trait}" & cs_id == {cs_id}'
        ).drop_duplicates('end')
        data.unique_finemap_df = finemap_df
        return data


class ComponentTracksLoader(PlotDataLoader):

    def _load(self, data: DataBundle, cutcounts_files, smooth=True, step=20, bandwidth=150, nmf_components=None):
        if not smooth:
            bandwidth = 1

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

# FIXME



class DHSIndexLoader(SegmentsLoader):
    __intervals_attr__ = 'dhs_intervals'
    
    def _load(self, data: DataBundle, dhs_index: pd.DataFrame, extra_columns=None, rectprops_columns=None):
        return super()._load(
            data, 
            segments_df=dhs_index, 
            extra_columns=extra_columns,
            rectprops_columns=rectprops_columns
        )


class FootprintsLoader(SegmentsLoader):
    __intervals_attr__ = 'footprint_intervals'
    def _load(self, data: DataBundle, footprints_index: pd.DataFrame, extra_columns=None, rectprops_columns=None):
        return super()._load(
            data, 
            segments_df=footprints_index, 
            extra_columns=extra_columns,
            rectprops_columns=rectprops_columns
        )


class DHSLoadingsLoader(PlotDataLoader):
    required_loader_kwargs = ['H']



class MotifLoader(PlotDataLoader):

    def _load(self, data: DataBundle, motif_annotations_path, motif_meta):
        interval_motif_annotations = TabixExtractor(motif_annotations_path,
                                                    columns=[
                                                        'chrom', 'start', 'end', 'fp_id',
                                                        'motif_chr', 'motif_start', 'motif_end',
                                                        'pfm', 'dg', 'orient', 'sequence'
                                                        ])[data.interval]
        interval_motif_annotations['dg'] = interval_motif_annotations['dg'].astype(float)
        interval_motif_annotations['start'] = interval_motif_annotations['start'].astype(int)
        interval_motif_annotations['end'] = interval_motif_annotations['end'].astype(int)
        interval_motif_annotations['motif_start'] = interval_motif_annotations['motif_start'].astype(int)
        interval_motif_annotations['motif_end'] = interval_motif_annotations['motif_end'].astype(int)
        interval_motif_annotations = interval_motif_annotations.groupby('fp_id', group_keys=False).apply(lambda x: x.nlargest(1, 'dg'))
        interval_motif_annotations['motif_id'] = interval_motif_annotations['pfm'].str.replace('.pfm', '')
        interval_motif_annotations = interval_motif_annotations.merge(motif_meta, left_on='motif_id', right_index=True)
        data.motif_intervals = df_to_genomic_intervals(
            interval_motif_annotations,
            data.interval,
            extra_columns=['orient', 'motif_start', 'motif_end', 'tf_name', 'pwm']
        )
        return data


class AggregatedCAVLoader(PlotDataLoader):

    def _load(self, data: DataBundle, cavs_data, fdr_tr=0.1, color='k', notsignif_color='#C0C0C0'):
        filtered_cavs = filter_df_to_interval(cavs_data, data.interval)
        filtered_cavs['is_significant'] = filtered_cavs['min_fdr'] <= fdr_tr
        filtered_cavs['sig_es'] = np.clip(np.where(filtered_cavs['is_significant'], np.abs(filtered_cavs['logit_es_combined']), 0), 0, 2)
        group_ids_df = filtered_cavs.query('is_significant').groupby(['#chr', 'start', 'end', 'ref', 'alt'])['group_id'].apply(lambda x: ','.join(map(str, x))).reset_index()
        filtered_cavs = filtered_cavs.groupby(['#chr', 'start', 'end', 'ref', 'alt'], group_keys=False).apply(lambda x: x.nlargest(1, 'sig_es'))
        filtered_cavs = filtered_cavs.merge(group_ids_df, on=['#chr', 'start', 'end', 'ref', 'alt'], how='left', suffixes=('', '_list'))
        
        filtered_cavs['value'] = np.abs(filtered_cavs['logit_es_combined'])
        filtered_cavs['color'] = np.where(filtered_cavs['is_significant'], color, notsignif_color)
        data.cavs_intervals = df_to_variant_intervals(filtered_cavs, extra_columns=['value', 'color'])
        return data
    

class PerSampleCAVLoader(PlotDataLoader):
    __required_fields__ = ['nonaggregated_cavs_data']

    def _load(self, data: DataBundle, nonaggregated_cavs_data, sample_id, fdr_tr=0.1, color='k', notsignif_color='#C0C0C0'):
        filtered_cavs = TabixExtractor(nonaggregated_cavs_data)[data.interval].query(f'sample_id == "{sample_id}"')
        filtered_cavs['is_significant'] = filtered_cavs['FDR_sample'] <= fdr_tr
        filtered_cavs['sig_es'] = np.clip(np.where(filtered_cavs['is_significant'], np.abs(filtered_cavs['logit_es']), 0), 0, 2)
        
        filtered_cavs['value'] = np.abs(filtered_cavs['logit_es'])
        filtered_cavs['color'] = np.where(filtered_cavs['is_significant'], color, notsignif_color)
        data.cavs_intervals = df_to_variant_intervals(filtered_cavs, extra_columns=['value', 'color'])
        return data


