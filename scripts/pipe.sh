#!/bin/bash
shopt -s expand_aliases

COMPONENTS=3
K_HIGH=11
MAF=0.05
create_env=false
while getopts ":p:k:m:v:" opt; do
    case ${opt} in
        p )
            COMPONENTS=$OPTARG
            ;;
        k )
            K_HIGH=$OPTARG
            ;;
        m )
            MAF=$OPTARG
            ;;
        v ) 
            create_env="yes"
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            exit 1
            ;;
        : )
            echo "Option -$OPTARG requires an argument." 1>&2
            exit 1
            ;;
    esac
done

repo_dir=$(realpath .)
dir_name=$(basename $repo_dir)
#if user is in scripts dir
if [ "$dir_name" == "scripts" ]; then
    echo "user is in scripts dir, switching to proj root"
    cd ..
    repo_dir=$(realpath .)
fi

if [ "$create_env" == "yes" ]; then
    echo "creating analysis environment..."
    conda env create --prefix ${repo_dir}/snp-clustering-env -f ${repo_dir}/snp-clustering-env.yml --yes
    eval "$(conda shell.bash hook)"
    conda activate ${repo_dir}/snp-clustering-env
    echo "done"
else
    eval "$(conda shell.bash hook)"
    conda activate snp-clustering-env
fi

alias plink="${repo_dir}/plink/plink"

data_dir=${repo_dir}/data
notebook_dir=${repo_dir}/notebooks
tmp_dir=${repo_dir}/tmp
out_dir=${repo_dir}/out

mkdir -p $tmp_dir $out_dir

VCF=${data_dir}/1000G_chr19_pruned.vcf.gz
VCF_TEST=${data_dir}/ps3_gwas.vcf.gz
PCA_OUT=clustering_pca

echo "using ${COMPONENTS} principal components"
echo "using at most ${K_HIGH} clusters"
echo "using at most ${K_HIGH} clusters"
echo "filtering out SNPs with MAF less than ${MAF}"
echo "writing plink intermediate data to ${tmp_dir}"


# to avoid curse of dimensionality, get top 3 principal components with plink
echo "getting top PCAs for clustering..."
plink --vcf $VCF --maf $MAF --make-bed --allow-no-sex --pca var-wts $COMPONENTS --out ${tmp_dir}/${PCA_OUT}
echo "done"

#execute notebook
echo "executing cluster analysis..."
papermill ${notebook_dir}/snp_clustering.ipynb ${out_dir}/snp_clustering_report.ipynb --cwd ${repo_dir}/notebooks -p DATA_PATH ${tmp_dir}/${PCA_OUT}.eigenvec -p K_HIGH 12
echo "done"

#convert to markdown
echo "exporting executed notebook to markdown..."
jupyter nbconvert --to markdown ${out_dir}/snp_clustering_report.ipynb --output-dir ${out_dir}
echo "whole analysis done"

conda deactivate