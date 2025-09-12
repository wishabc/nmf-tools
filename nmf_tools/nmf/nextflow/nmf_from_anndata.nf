#!/usr/bin/env nextflow
nextflow.enable.dsl = 2


process fit_nmf {
	tag "${prefix}"
	conda params.conda
    publishDir "${params.outdir}/nmf/${prefix}", pattern: "${prefix}.*"
    label "highmem"

	input:
		val prefix

	output:
        tuple val(prefix), path("${prefix}.W.npy"), path("${prefix}.H.npy"), path("${prefix}.non_zero_peaks_mask.txt")

	script:
	"""
    python3 $moduleDir/bin/nmf/run_NMF.py \
        ${prefix} \
        ${params.nmf_config}
	"""
}

process visualize_nmf {
	tag "${prefix}"
	conda params.conda
    publishDir "${params.outdir}/nmf/${prefix}"
    label "highmem"
    errorStrategy 'ignore'

	input:
        tuple val(prefix), path(W), path(H), path(non_zero_peaks_mask)

	output:
        tuple val(prefix), path("*.pdf")

	script:
	"""
    python3 $moduleDir/bin/nmf/visualize_nmf.py \
        ${prefix} \
        ${params.nmf_config} \
        ${W} \
        ${H}
	"""
}

process add_metadata {
    conda params.conda
    publishDir "${params.outdir}"
    errorStrategy 'ignore'

    output:
        path name

    script:
    name = "nmf_meta+matrices.tsv"
    """
    python3 $moduleDir/bin/nmf/add_metadata.py \
        ${params.nmf_config} \
        ${params.outdir}/nmf \
        ${name}
    """
}


// nextflow run ~/packages/nmf_tools/nmf/nextflow/nmf_from_anndata.nf -profile Altius -resume
workflow {
    Channel.fromPath(params.nmf_params)
        | splitCsv(header: false)
        | distinct { it[0] }
        | map(it -> it[0])
        | fit_nmf
        | visualize_nmf

    add_metadata()
}


// Entry for visuzizations only
workflow visualize {
    println "Visualizing NMF results from params.nmf_params_list = ${params.nmf_params_list}"
    Channel.fromPath(params.nmf_params_list)
        | splitCsv(header:true, sep:'\t')
		| map(row -> tuple(
            row.prefix,
            row.n_components,
            file(row.anndata_path),
            ))
        | distinct { it[0] }
        | map( 
            it -> tuple(
                *it[0..(it.size()-1)],
                file("${params.nmf_results_path}/${it[0]}/${it[0]}.W.npy"),
                file("${params.nmf_results_path}/${it[0]}/${it[0]}.H.npy"),
                file("${params.nmf_results_path}/${it[0]}/${it[0]}.non_zero_peaks_mask.txt"),
            )
        )
        | filter { it[3].exists() }
        | visualize_nmf
}
