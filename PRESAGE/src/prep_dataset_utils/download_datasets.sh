#mkdir ../data/raw_data/
#cd ../data/raw_data/
#wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01.h5ad
#wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01.h5ad

#cd ..
#echo "replogle_k562_gw"
#mkdir replogle_k562_gw
#wget -c -O replogle_k562_gw/perturb_processed.h5ad "https://zenodo.org/records/7041849/files/ReplogleWeissman2022_K562_gwps.h5ad?download=1"
#python3 ../src/prep_dataset_utils/TransformscPerturbdatasets.py replogle_k562_gw

cd ../data
echo "replogle_k562_essential_unfiltered"
mkdir replogle_k562_essential_unfiltered
wget -c -O replogle_k562_essential_unfiltered/perturb_processed.h5ad "https://zenodo.org/records/7041849/files/ReplogleWeissman2022_K562_essential.h5ad?download=1"
python3 ../src/prep_dataset_utils/TransformscPerturbdatasets.py replogle_k562_essential_unfiltered

#echo "replogle_rpe1_essential_unfiltered"
#mkdir replogle_rpe1_essential_unfiltered
#wget -c -O replogle_rpe1_essential_unfiltered//perturb_processed.h5ad "https://zenodo.org/records/7041849/files/ReplogleWeissman2022_rpe1.h5ad?download=1"
#python3 ../src/prep_dataset_utils/TransformscPerturbdatasets.py replogle_rpe1_essential_unfiltered

cd ..
#./src/prep_dataset_utils/prep_naidg.sh
