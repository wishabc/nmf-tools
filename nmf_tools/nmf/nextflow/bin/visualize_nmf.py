import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import scipy.cluster.hierarchy as sch

from args_parser import NMFInputData, setup_parser, parse_nmf_args

from nmf_tools.utils import get_component_data
from nmf_tools.plotting.matrices_barplots import component_barplot, component_barplot_at_scale, plot_top_contributing_samples 


def main(
        nmf_data: NMFInputData,
        W: np.ndarray,
        H: np.ndarray,
        vis_path,
        component_data=None
    ):
    if component_data is None:
        component_data = get_component_data(W)

    print('Order samples by component contribution')
    project_masked_samples = nmf_data.project_masked_samples

    metadata = nmf_data.samples_metadata
    samples_mask = nmf_data.samples_mask
    peaks_mask = nmf_data.peaks_mask

    binary_matrix = nmf_data.matrix[peaks_mask, :]
    dhs_meta = nmf_data.dhs_metadata[peaks_mask]

    for i, row in component_data.iterrows():
        weights = np.ones(W.shape[0])
        weights[i] = W.shape[0]

        fig, _ = component_barplot_at_scale(
            W,
            component_data=component_data,
            records_labels=metadata['sample_label'].values,
            order_components_by='mean_loading',
            order_records_by='component',
            order_records_kwargs=dict(ind=i)
        )
        comp_name = row["name"].replace("/", ".").replace(' ', '.')
        plt.savefig(f'{vis_path}/Detailed_barplot.{comp_name}.pdf', transparent=True, bbox_inches='tight')
        plt.close(fig)



    if not project_masked_samples:
        metadata = metadata[samples_mask]

    ######### Plot samples #########
    if project_masked_samples:
        print('Reference samples set')
        ax, _, _ = component_barplot(W[:, samples_mask], component_data)
        plt.savefig(f'{vis_path}.Barplot_reference_train_samples.pdf', transparent=True, bbox_inches='tight')
        plt.close(ax.get_figure())
    

    print('All samples')
    ax, _, _ = component_barplot(W, component_data, order_records_by='primary')
    plt.savefig(f'{vis_path}.Barplot_all_samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(ax.get_figure())


    ######## Plot peaks #########
    print('All peaks')
    ax, _, _ = component_barplot(H, component_data)
    plt.savefig(f'{vis_path}.Barplot_all_DHSs.pdf', transparent=True, bbox_inches='tight')
    plt.close(ax.get_figure())

    print('All peaks not normalized')
    ax, _, _ = component_barplot(H, component_data, normalize=False)
    plt.savefig(f'{vis_path}.Barplot_all_DHSs.not_norm.pdf', transparent=True, bbox_inches='tight')
    plt.close(ax.get_figure())

    #Only reproduced DHSs
    print('>3 peaks supproting a DHS')
    reproduced_peaks = binary_matrix.sum(axis=1) > 3
    ax, _, _ = component_barplot(H[:, reproduced_peaks], component_data, normalize=True)
    plt.savefig(f'{vis_path}.Barplot_DHS_supported_by_4+samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(ax.get_figure())

    print('Detailed barplot all samples')
    s_order, fig = component_barplot_at_scale(W, metadata, colors=component_data['color'])
    if project_masked_samples:
        plt.close(fig)
        s_mask = samples_mask[s_order]
        component_barplot_at_scale(
            W,
            metadata,
            colors=component_data['color'],
            order=s_order,
            label_colors=[
                'r' if s else 'k' for s in s_mask
            ]
        )
    plt.savefig(f'{vis_path}.Detailed_barplot_all_samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

    print('Hierarchical barplot all samples')
    _, fig = component_barplot_at_scale(
        W,
        metadata,
        colors=component_data['color'],
        order=s_order
    )
    if project_masked_samples:
        plt.close(fig)
        s_mask = samples_mask[s_order]
        component_barplot_at_scale(
            W,
            component_data,
            order_records_by='hierarchical',
            order_records_kwargs=dict(optimal_ordering=False),
            normalize_for_plotting=False,
            colors=component_data['color'],
            order=s_order,
            label_colors=[
                'r' if s else 'k' for s in s_mask
            ]
        )
    plt.savefig(f'{vis_path}.Hierarchical_barplot_all_samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

    print('Hierarchical barplot reference samples')
    if samples_mask.sum() < samples_mask.shape[0] and project_masked_samples:
        _, fig = component_barplot_at_scale(
            W[:, samples_mask],
            metadata.loc[samples_mask, :],
            colors=component_data['color'],
            order=s_order
        )
        plt.savefig(f'{vis_path}.Hierarchical_barplot_reference_samples.pdf', transparent=True, bbox_inches='tight')
        plt.close(fig)

    print('Top 20 samples per component')
    annotations = metadata["sample_label"].values
    fig, axes = plot_top_contributing_samples(W, annotations, top_count=20, component_data=component_data)
    plt.savefig(f'{vis_path}.Top20_all_samples_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)

    fig, axes = plot_top_contributing_samples(W, annotations, top_count=20, component_data=component_data, common_scale=True)
    plt.savefig(f'{vis_path}.Top20_all_samples_barplot.common_scale.pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)

    if project_masked_samples:
        fig, axes = plot_top_contributing_samples(W[:, samples_mask], annotations[samples_mask], top_count=20, component_data=component_data)
        plt.savefig(f'{vis_path}.Top20_reference_samples_barplot.pdf', bbox_inches='tight', transparent=True)
        plt.close(fig)

        fig, axes = plot_top_contributing_samples(W[:, samples_mask], annotations[samples_mask], top_count=20, component_data=component_data, common_scale=True)
        plt.savefig(f'{vis_path}.Top20_reference_samples_barplot.common_scale.pdf', bbox_inches='tight', transparent=True)
        plt.close(fig)


    if 'dist_tss' in dhs_meta.columns:
        ax = plot_dist_tss(H, dhs_meta['dist_tss'], component_data)
        plt.savefig(f'{vis_path}.Distance_to_tss.pdf', bbox_inches='tight', transparent=True)
        plt.close(fig)



def plot_dist_tss(H, dist_tss, component_data, ax=None):
    max_component = np.argmax(H, axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))
    for i, row in component_data.iterrows():
        data = np.abs(dist_tss[max_component == row['index']])
        ax.plot(np.sort(data), np.linspace(0, 1, len(data)), color=row['color'])
    ax.set_xlim(-50, 5000)
    ax.set_xlabel('Distance to TSS')
    ax.set_ylabel('Cumulative proportion of DHSs')
    return ax


if __name__ == '__main__':
    print('Visualizing NMF results')
    parser = setup_parser()

    print('Adding options to parser')
    parser.add_argument('W', help='W matrix of perform NMF decomposition')
    parser.add_argument('H', help='H matrix of perform NMF decomposition')
    parser.add_argument('--outpath', help='Path to save visualizations', default='./')
    args = parser.parse_args()

    nmf_data = parse_nmf_args(args.prefix, args.config)
    
    if 'sample_label' not in nmf_data.samples_metadata.columns:
        if 'SPOT3_score' not in nmf_data.samples_metadata.columns and 'core_ontology_term' not in nmf_data.samples_metadata.columns:
            raise ValueError('sample_label column is missing in samples metadata')
        print('Assuming sample_label is core_ontology_term + SPOT3_score')
        nmf_data.samples_metadata.loc[:, 'sample_label'] = (
            nmf_data.samples_metadata['core_ontology_term'].astype(str) + " " + 
            nmf_data.samples_metadata['SPOT3_score'].astype(str).str[:3]
        )

    W = np.load(args.W).T # NMF components x samples
    H = np.load(args.H).T # NMF components x peaks
    outprefix = f"{args.outpath}/{args.prefix}"
    main(nmf_data, W, H, outprefix)

