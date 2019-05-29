
#!/bin/sh 
echo Running all 	 

 
for t in 4 #4 8 16
	do
		for x in  500 #10000 20000      

		do
 
			python ptBayeslands_extended_novisualize.py -p 2 -s $x -r 10 -t 5 -swap 0.01 -b 0.25 -pt 0.5 -epsilon 0.5 -rain_intervals $t
 
		done
	done 

