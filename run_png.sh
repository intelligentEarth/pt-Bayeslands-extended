
#!/bin/sh 
echo Running all 	 

 
for prob in 6
	do
		for run in 1
		do
			python ptBayeslands_sedvec.py -p $prob -s 10000 -r 10 -t 10 -swap 0.1 -b 0.25 -pt 0.5
 
		done
	done 


 