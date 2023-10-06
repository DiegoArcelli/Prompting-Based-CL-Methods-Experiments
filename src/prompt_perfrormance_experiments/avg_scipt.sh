cat output_multiple_accuracy | grep "Total accuracy:" | head -n 252 | cut -d ":" -f2 | awk '{ total += $1 } END { if (NR > 0) print "Average:", total / NR }' > avg_prompt_accuracy
