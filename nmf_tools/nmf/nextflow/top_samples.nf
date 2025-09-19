#!/usr/bin/env nextflow
nextflow.enable.dsl = 2


process find_top_samples {

    conda params.conda
    tag "${prefix}"
    publishDir "${params.outdir}/top_samples", pattern: "${name}"
    publishDir "${params.outdir}/top_samples", pattern: "${res}"

    input:
        tuple val(prefix), path(W), path(H), path(non_zero_peaks_mask)

    output:
        tuple path("*.*.component_${prefix}.bw"), path(name), path(res)

    script:
    name = "${prefix}.top_samples.tsv"
    res = "${prefix}.density_tracks_meta.tsv"
    """
    python3 $moduleDir/bin/find_top_samples.py \
        ${prefix} \
        ${params.nmf_config} \
        ${W_matrix} \
        ${non_zero_peaks_mask} \
        ${params.top_count} \
        ${params.outdir}/top_samples/${prefix} \
    """
}


process top_samples_track {

    scratch true
    conda params.conda
    tag "${prefix}:${component}"
    publishDir "${params.outdir}/top_samples/${prefix}"

    input:
        tuple val(component), val(prefix), path(density_bw, stageAs: "?/*")
    
    output:
        tuple val(prefix), path(name), path(bg)
    
    script:
    name = "${prefix}.${component}.top_samples.bw"
    bg = "${prefix}.${component}.top_samples.bg"
    """
    wiggletools write_bg ${bg} mean ${density_bw}
    bedGraphToBigWig "${bg}" "${params.chrom_sizes}" "${name}"
    """
}

workflow findTop {
    take:
        data // 
    main:
        data
            | find_top_samples
            | map(it -> it[0])
            | flatten()
            | combine(nmf_data.map(it -> it[0]))
            | map(it -> tuple(it[0].simpleName, it[1], it[0]))
            | groupTuple(by: [0, 1])
            | top_samples_track

    emit:

}