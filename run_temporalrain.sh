
#!/bin/sh 
echo Running all 	 

 
for t in 4 8 16
	do
		for x in  5000 10000   20000 40000    # 4 8 16

		do
 
			python ptBayeslands_extended.py -p 2 -s $x -r 10 -t 10 -swap 0.1 -b 0.25 -pt 0.5 -epsilon 0.5 -rain_intervals $t
 
		done
	done 

