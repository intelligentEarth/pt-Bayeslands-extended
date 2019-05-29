
#!/bin/sh 
echo Running all 	 

 
for t in 1   #4 8 16
	do      
		#sleep 1m
 
			python realtime_visualise_results.py -p 2 -s 1000 -r 8 -t 10 -swap 0.01 -b 0.25 -pt 0.5 -epsilon 0.5 -rain_intervals 4

  
	done 

