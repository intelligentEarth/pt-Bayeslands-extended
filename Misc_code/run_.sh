
#!/bin/sh 
echo Running all 	 

 
for prob in 1
	do
		for x in 0.4 0.6  0.8 

		do

			#let "ep = $x/10." 
			python ptBayeslands_extended__.py -p 3 -s 20000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5 -epsilon $x

			python ptBayeslands_extended__.py -p 3 -s 40000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5 -epsilon $x
 
		done
	done 


 