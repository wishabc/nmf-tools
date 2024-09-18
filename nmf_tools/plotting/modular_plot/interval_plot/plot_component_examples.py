from matplotlib.offsetbox import TextArea, VPacker, HPacker, AnnotationBbox

from nmf_tools.plotting.modular_plot import uses_loaders
from nmf_tools.plotting.modular_plot.interval_plot.plot_components import VerticalPlotComponent, SingleBPObjectsComponent, SignalPlotComponent

import numpy as np

from .loaders import *

from matplotlib import gridspec
import matplotlib.pyplot as plt

from genome_tools.plotting import signal_plot, segment_plot
from genome_tools.plotting.gene_annotation import gene_annotation_plot
from genome_tools.plotting.ideogram import ideogram_plot
from genome_tools.plotting.utils import clear_spines
from genome_tools.plotting.pwm import plot_motif_logo
from genome_tools.plotting.colors.cm import get_vocab_color

from nmf_tools import in_vierstra_style
from nmf_tools.plotting.matrices_barplots import component_barplot


@uses_loaders(IdeogramLoader)
class IdeogramComponent(VerticalPlotComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @in_vierstra_style
    def _plot(self, data, ax, **kwargs):
        ideogram_plot(data.ideogram_data, data.interval.chrom, pos=data.interval.start, ax=ax, **kwargs)
        return ax


@uses_loaders(GencodeLoader)
class GencodeComponent(VerticalPlotComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        try:
            gene_annotation_plot(data.interval, data.gencode_annotation_file, ax=ax,
                                 gene_symbol_exclude_regex=r'^ENSG|^MIR|^LINC|.*-AS.*',
                                 **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(data.interval.start, data.interval.end)
            clear_spines(ax)
        except ValueError:
            self.logger.warning("No gene annotations found for the interval.")
        return ax


@uses_loaders(FinemapLoader)
class FinemapComponent(SingleBPObjectsComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _plot(self, data, ax, **kwargs):
        data.positions = data.unique_finemap_df['start'] + 0.5
        data.values = data.unique_finemap_df['pip']
        return super().plot(data, ax, **kwargs)
    

@uses_loaders(SignalLoader)
class TrackComponent(VerticalPlotComponent):
    def __init__(self, signal_files, smooth=True, step=10, bandwidth=150, **kwargs):
        super().__init__(**kwargs)
        self.loader_kws.update(dict(
            step=step,
            bandwidth=bandwidth,
            signal_files=signal_files,
            smooth=smooth,
        ))

    def _plot(self, data, ax, **kwargs):
        ax.set_xlim(data.interval.start, data.interval.end)
        signal_plot(data.interval, data.signal, ax=ax, **kwargs)
        return ax


@uses_loaders(ComponentTracksLoader)
class NMFTracksComponent(SignalPlotComponent):
    def __init__(self, component_data=None, nmf_components=None, **kwargs):
        if component_data is None:
            raise ValueError("Component data must be provided for DNaseTracksComponent.")
        super().__init__(**kwargs)
        self.component_data = component_data
        if nmf_components is not None:
            self.loader_kws.update({'nmf_components': nmf_components})

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        density_axes = self.plot_component_tracks(data.interval, data.nmf_components,
                                                  data.component_tracks, self.component_data,
                                                  gridspec_ax=ax, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("DNase I\ndensity")
        clear_spines(ax)
        return density_axes, ax
    

    @staticmethod
    def plot_component_tracks(interval, components, component_tracks, component_data,
                              gridspec_ax, common_lim=False, **kwargs):
        assert len(components) == len(component_tracks)
        gss = gridspec.GridSpecFromSubplotSpec(len(components), 1, subplot_spec=gridspec_ax, hspace=0.05)
        axes = []
        colors = component_data.set_index('index').loc[components, 'color'].values
        labels = component_data.set_index('index').loc[components, 'name'].values
        lims = []
        for i, (color, label, segs) in enumerate(zip(colors, labels, component_tracks)):
            ax_dens = gridspec_ax.get_figure().add_subplot(gss[i])
            lim = segs.max()
            lims.append(lim)
            signal_plot(interval, segs, ax=ax_dens, color=color, lw=0, **kwargs)
            ax_dens.set_ylim(0, lim)
            ax_dens.set_yticks([])
            if i != len(components) - 1:
                ax_dens.set_xticks([])
            ax_dens.text(x=0.005, y=0.45, ha='left', va='bottom', s=label, transform=ax_dens.transAxes, fontsize=5)
            axes.append(ax_dens)

        if common_lim:
            for ax in axes:
                ax.set_ylim(0, max(lims))
        return axes


@uses_loaders(DHSIndexLoader)
class DHSIndexComponent(VerticalPlotComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        segment_plot(data.interval, data.dhs_intervals, ax=ax, **kwargs)
        ax.set_xlim(data.interval.start, data.interval.end)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('NMF-annotated\nDHSs')
        clear_spines(ax)
        return ax


@uses_loaders(DHSIndexLoader, DHSLoadingsLoader)
class DHSLoadingsComponent(VerticalPlotComponent):
    def __init__(self, component_data=None, **kwargs):
        if component_data is None:
            raise ValueError("Component data must be provided for DHSLoadingsComponent.")
        super().__init__(**kwargs)
        self.component_data = component_data

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        ax.axis('off')
        axes = self.add_axes_at_middle_points(data.dhs_intervals, data.interval, ax=ax, bp_width=50)
        self.plot_barplots_for_dhs(data.dhs_intervals, axes, H=data.H, component_data=self.component_data)
        return axes, ax
    
    @staticmethod
    def plot_barplots_for_dhs(genomic_intervals, axes, H, component_data):
        assert len(genomic_intervals) == len(axes)
        for genomic_interval, ax in zip(genomic_intervals, axes):
            component_barplot(H[:, genomic_interval.index: genomic_interval.index + 1], component_data, ax=ax, normalize=True)


#TODO: unify with DHSIndexComponent
@uses_loaders(FootprintsLoader)
class FootprintsComponent(VerticalPlotComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        segment_plot(data.interval, data.footprint_intervals, ax=ax, color='k')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Footprint\nindex')
        clear_spines(ax)
        return ax


@uses_loaders(FootprintDatasetLoader)
class FootprintTrackComponent(VerticalPlotComponent):
    def __init__(self, fp_samples, kind='pp', fdr_cutoff=0.05, **kwargs):
        super().__init__(**kwargs)
        self.loader_kws.update({
            'fp_samples': fp_samples,
            'fdr_cutoff': fdr_cutoff,
            })
        assert kind in ['pp', 'obs/exp']
        self.kind = kind

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, smpl_idx=0, color='k', exp_color='C1', lw=0.5, **kwargs):
        xs = self.squarify_array(np.arange(data.pp.shape[1] + 1) + data.interval.start)
        if self.kind == 'pp':
            ax.plot(xs, np.repeat(data.pp[smpl_idx, :], 2), color=color, lw=lw, **kwargs)
        elif self.kind == 'obs/exp':
            ax.plot(xs, np.repeat(data.obs[smpl_idx, :], 2), color=exp_color, lw=lw, **kwargs)
            ax.plot(xs, np.repeat(data.exp[smpl_idx, :], 2), color=color, lw=lw, **kwargs)
        ax.set_xticks([])
        return ax
    
    @staticmethod
    def squarify_array(y):
        return np.concatenate([y[:1], np.repeat(y[1:-1], 2), y[-1:]])


@uses_loaders(MotifLoader)
class MotifComponent(VerticalPlotComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        ax.axis('off')
        axes = self.add_axes_at_middle_points(data.motif_intervals, data.interval, ax=ax)
        self.plot_motifs_for_footprints(data.motif_intervals, axes)
        return ax
    
    @staticmethod
    def plot_motifs_for_footprints(motif_intervals, axes):
        assert len(motif_intervals) == len(axes)
        for fp_interval, ax in zip(motif_intervals, axes):
            plot_motif_logo(fp_interval.pwm, rc=fp_interval.orient == '-', font='IBM Plex Mono', ax=ax)
            ax.set_xlabel(fp_interval.tf_name, labelpad=0.5)


@uses_loaders(CAVLoader)
class CAVComponent(SingleBPObjectsComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        abs_es = np.abs(data.cavs['logit_es_combined'])
        self.plot_single_bp_objects(
            data.cavs['start'] + 0.5,
            abs_es,
            data.interval,
            ax=ax,
            colors=np.where(data.cavs['min_fdr'] <= 0.1, 'k', '#C0C0C0')
        )
        self.annotate_cav_alleles(data.cavs, ax=ax)
        ax.set_ylim(0, max(abs_es.max() * 1.2, 2.0))
        ax.set_xticks([])
        ax.set_ylabel("CAV\neffect size")
        return ax
    
    @staticmethod
    def annotate_cav_alleles(df, ax=None, box_alignment=(0.5, -0.6), **kwargs):
        if ax is None:
            ax = plt.gca()

        for _, row in df.iterrows():
            x = row['start'] + 0.5
            y = row['sig_es']
            a1, a2 = row['ref'], row['alt']
            
            if row['es_combined'] < 0:
                a1, a2 = a2, a1
            
            text_areas = []
            
            for text, color in zip([a1, '/', a2], [get_vocab_color(a1, 'dna'), 'k', get_vocab_color(a2, 'dna')]):
                text_area = TextArea(text, textprops=dict(color=color, fontsize=5, ha='center', va='center'))
                text_areas.append(text_area)
            
            hp = HPacker(children=text_areas, align="bottom", pad=0, sep=1)
            ab = AnnotationBbox(hp, (x, y), frameon=False, box_alignment=box_alignment, xycoords='data', **kwargs)
            ax.add_artist(ab)
            
        return ax
