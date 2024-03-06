repo_dir=$(realpath .)
dir_name=$(basename $repo_dir)
#if user is in scripts dir
if [ "$dir_name" == "scripts" ]; then
    echo "user is in scripts dir, switching to proj root"
    cd ..
    repo_dir=$(realpath .)
fi

alias plink="${repo_dir}/plink/plink"

data_dir=${repo_dir}/data
notebook_dir=${repo_dir}/notebooks
tmp_dir=${repo_dir}/tmp
out_dir=${repo_dir}/out

echo "cleaning tmp and output directories"
rm -rf ${tmp_dir}/*
rm -rf ${out_dir}/*

echo "done"