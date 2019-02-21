
#!/bin/sh 
echo Running all 	 

 
for prob in 3
	do
		for run in {2..10..2}

		do

			let "samples = $run * 10000"
			echo $samples
			python ptBayeslands_extended.py -p $prob -s $samples -r 10 -t 10 -swap 0.1 -b 0.25 -pt 0.5
 
		done
	done 


 