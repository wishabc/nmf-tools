import pandas as pd


from genome_tools.data.extractors import TabixExtractor, FastaExtractor, VariantGenotypeExtractor
from genome_tools.utils.signal import smooth_and_aggregate_per_nucleotide_signal
from genome_tools import df_to_genomic_intervals, VariantInterval

from nmf_tools.plotting.modular_plot.interval_plot.loaders import PlotDataLoader, DataBundle

from nmf_tools.plotting.extract_reads import extract_allelic_reads


class IdeogramLoader(PlotDataLoader):
    required_loader_kwargs = ['ideogram_data']


class GencodeLoader(PlotDataLoader):
    required_loader_kwargs = ['gencode_annotation_file']



class SignalLoader(PlotDataLoader):
    
    def _load(self, data, signal_files, smooth=True, step=20, bandwidth=150):
        if not smooth:
            bandwidth = 1
        segs = smooth_and_aggregate_per_nucleotide_signal(data.interval, signal_files,
                                                          step=step, bandwidth=bandwidth)
        data.signal = segs
        return data


class SegmentsLoader(PlotDataLoader):
    __intervals_attr__ = 'intervals'

    def _load(self, data: DataBundle, segments_df: pd.DataFrame, extra_columns=None, rectprops_columns=None):
        if rectprops_columns is None:
            rectprops_columns = []
        if extra_columns is None:
            extra_columns = []
        setattr(
            data,
            self.__intervals_attr__, 
            df_to_genomic_intervals(
                segments_df.reset_index(drop=True).reset_index(),
                data.interval,
                extra_columns=['index'] + extra_columns + rectprops_columns
            )
        )

        if rectprops_columns:
            for interval in getattr(data, self.__intervals_attr__):
                interval.rectprops = {
                    col: getattr(interval, col) for col in rectprops_columns
                    }
        return data



class AllelicReadsLoader(PlotDataLoader):

    def _load(self, data: DataBundle, sample_ids, samples_metadata: pd.DataFrame, variant_interval: VariantInterval):
        if isinstance(sample_ids, (str, int, float)):
            sample_ids = [sample_ids]
        cram_paths = samples_metadata.loc[sample_ids, 'cram_file']
        reads = {}
        for sample_id, cram_path in zip(sample_ids, cram_paths):
            reads[sample_id] = extract_allelic_reads(cram_path, variant_interval, data.interval)
        data.reads = reads
        return data


class FastaLoader(PlotDataLoader):
    def _load(self, data: DataBundle, fasta_file):
        with FastaExtractor(fasta_file) as ext:
            data.sequence = ext[data.interval]
        return data


class VariantGenotypeLoader(PlotDataLoader):
    
    def _load(self, data: DataBundle, vcf_path):
        with VariantGenotypeExtractor(vcf_path) as extractor:
            variants = extractor[data.interval].rename(columns={'pos': 'end'})
        
        gt_mapping = {
            (0, 0): "A",
            (1, 1): "B",
        }
        
        variants["parsed_genotype"] = variants["gt"].map(gt_mapping)
        variants = variants.dropna(subset=["parsed_genotype"])
        

        variants['start'] = variants['end'] + 1
        data.variant_genotype = variants
        return data
