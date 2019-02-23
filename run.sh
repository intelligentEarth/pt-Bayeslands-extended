
#!/bin/sh 
echo Running all 	 

 
for prob in 3
	do
		for run in {1..3..1}

		do

			#let "samples = $run * 1000"

			echo $samples
			python ptBayeslands_extended.py -p $prob -s 20000 -r 10 -t 10 -swap 0.1 -b 0.25 -pt 0.5
 
		done
	done 


 