set terminal png size 900,600
set output 'cost_plot.png'
set title "Évolution du coût durant l'entraînement"
set xlabel "Itération"
set ylabel "Coût"
set grid
plot "cost_history.dat" using 1:2 with lines lw 2 lc rgb "blue" title "Coût"
