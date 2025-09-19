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
        tuple val(prefix), path("*.*.component_${prefix}.bw"), path(name), path(res)

    script:
    name = "${prefix}.top_samples.tsv"
    res = "${prefix}.density_tracks_meta.tsv"
    """
    python3 $moduleDir/bin/find_top_samples.py \
        ${prefix} \
        ${params.nmf_config} \
        ${W} \
        ${H} \
        ${non_zero_peaks_mask} \
        ${params.top_count}
    """
}


process top_samples_track {

    scratch true
    conda params.conda
    tag "${prefix}:${component}"
    publishDir "${params.outdir}/top_samples/${prefix}"

    input:
        tuple val(prefix), val(component), path(density_bw, stageAs: "?/*")
    
    output:
        tuple val(prefix), val(component), path(name)
    
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
        top_samples = data
            | find_top_samples
            | map(it -> tuple(it[0], it[1]))
            | transpose() // prefix, bw_file

            | map(it -> tuple(it[0], it[1].simpleName, it[1]))
            | groupTuple(by: [0, 1]) // component, prefix, bw_files
            | top_samples_track // 
            | collectFile(
                storeDir: "${params.outdir}/top_samples",
                skip: 1,
                keepHeader: true
            ) {
                [
                    "${it[0]}.components_meta.tsv", //name
                    "component\tbw\n${it[1]}\t${params.outdir}/top_samples/${it[0]}/${it[2].name}" // content
                ]
            }

    emit:
        top_samples
}