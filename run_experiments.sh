mkdir results/

python experiments.py wine gnb logreg
python experiments.py wine knn logreg
python experiments.py wine randomforest logreg
python analyze_results.py results/results_wine_gnb_logreg.npz
python analyze_results.py results/results_wine_knn_logreg.npz
python analyze_results.py results/results_wine_randomforest_logreg.npz

python experiments.py wine gnb dectree
python experiments.py wine knn dectree
python experiments.py wine randomforest dectree
python analyze_results.py results/results_wine_gnb_dectree.npz
python analyze_results.py results/results_wine_knn_dectree.npz
python analyze_results.py results/results_wine_randomforest_dectree.npz

python experiments.py breastcancer gnb logreg
python experiments.py breastcancer knn logreg
python experiments.py breastcancer randomforest logreg
python analyze_results.py results/results_breastcancer_gnb_logreg.npz
python analyze_results.py results/results_breastcancer_knn_logreg.npz
python analyze_results.py results/results_breastcancer_randomforest_logreg.npz

python experiments.py breastcancer gnb dectree
python experiments.py breastcancer knn dectree
python experiments.py breastcancer randomforest dectree
python analyze_results.py results/results_breastcancer_gnb_dectree.npz
python analyze_results.py results/results_breastcancer_knn_dectree.npz
python analyze_results.py results/results_breastcancer_randomforest_dectree.npz

python experiments.py t21 gnb logreg
python experiments.py t21 knn logreg
python experiments.py t21 randomforest logreg
python analyze_results.py results/results_t21_gnb_logreg.npz
python analyze_results.py results/results_t21_knn_logreg.npz
python analyze_results.py results/results_t21_randomforest_logreg.npz

python experiments.py t21 gnb dectree
python experiments.py t21 knn dectree
python experiments.py t21 randomforest dectree
python analyze_results.py results/results_t21_gnb_dectree.npz
python analyze_results.py results/results_t21_knn_dectree.npz
python analyze_results.py results/results_t21_randomforest_dectree.npz

python experiments.py flip gnb logreg
python experiments.py flip knn logreg
python experiments.py flip randomforest logreg
python analyze_results.py results/results_flip_gnb_logreg.npz
python analyze_results.py results/results_flip_knn_logreg.npz
python analyze_results.py results/results_flip_randomforest_logreg.npz

python experiments.py flip gnb dectree
python experiments.py flip knn dectree
python experiments.py flip randomforest dectree
python analyze_results.py results/results_flip_gnb_dectree.npz
python analyze_results.py results/results_flip_knn_dectree.npz
python analyze_results.py results/results_flip_randomforest_dectree.npz