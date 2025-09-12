import numpy as np

from nmf_tools.nmf import NMFModel

from args_parser import NMFInputData, setup_parser, parse_nmf_args


# Internal function to run NMF decomposition as part of pipeline
def main(nmf_input_data: NMFInputData):
    # Data preparation
    nmf_model = NMFModel(
        n_components=nmf_input_data.n_components,
        extra_params=nmf_input_data.extra_params,
    )

    if nmf_input_data.mode == "scaled_X":
        nmf_input_data.matrix = nmf_input_data.matrix.multiply(
            np.sqrt(nmf_input_data.peaks_weights)
        ).T.multiply(
            np.sqrt(nmf_input_data.samples_weights)
        ).T

    if nmf_input_data.samples_mask.sum() < nmf_input_data.samples_mask.shape[0]:
        X = nmf_input_data.matrix[nmf_input_data.samples_mask, :]
    else:
        X = nmf_input_data.matrix
    
    peaks_mask = nmf_input_data.peaks_mask & (X.sum(axis=0) > 0).A1
    X = X[:, peaks_mask]

    # Initial fit
    if nmf_input_data.mode == "scaled_X":
        W, H = nmf_model.fit_transform(X)
    else:
        W, H = nmf_model.fit_transform(
            X,
            W_weights=nmf_input_data.samples_weights[nmf_input_data.samples_mask],
            H_weights=nmf_input_data.peaks_weights[peaks_mask]
        )
    
    # Optional, projects masked samples
    if nmf_input_data.project_masked_samples:
        if nmf_input_data.mode == "scaled_X":
            W = nmf_model.project_samples(
                nmf_input_data.matrix[:, peaks_mask],
                H
            )
        else:
            W = nmf_model.project_samples(
                nmf_input_data.matrix[:, peaks_mask],
                H,
                W_weights=nmf_input_data.samples_weights,
                H_weights=nmf_input_data.peaks_weights[peaks_mask]
            )

    return W, H, peaks_mask


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()

    nmf_data = parse_nmf_args(args.prefix, args.config)

    W_np, H_np, peaks_mask = main(nmf_data)

    print('Saving results')
    np.save(f'{args.prefix}.W.npy', W_np) # samples x components
    np.save(f'{args.prefix}.H.npy', H_np.T) # peaks x components
    np.savetxt(f'{args.prefix}.non_zero_peaks_mask.txt', peaks_mask, fmt="%d")
    np.savetxt(f'{args.prefix}.samples_mask.txt', nmf_data.samples_mask, fmt="%d")