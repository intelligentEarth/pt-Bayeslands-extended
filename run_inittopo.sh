
#!/bin/sh 
echo Running all 	 

 
for prob in 1
	do
		for x in   1.0 0.8 0.6 0.4 0.2 

		do
  

			python ptBayeslands_extended.py -p 3 -s 10000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5 -epsilon $x

			#python ptBayeslands_extended.py -p 3 -s 20000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5 -epsilon $x

			 
		done
	done 


 