
#!/bin/sh 
echo Running all 	 

 
for prob in 1
	do
		for x in 0.2  

		do

			#let "ep = $x/10." 
			python ptBayeslands_extended_.py -p 3 -s 1000 -r 10 -t 10 -swap 0.1 -b 0.25 -pt 0.5 -epsilon $x
 
		done
	done 


 