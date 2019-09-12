### Task
	In this task, you will estimate a probability distribution using a Bayesian approach. 
	Again, the data will be a string S whose characters are only 'a' and 'b'. However:
	• This time the string will be provided to your program as a command line argument.
	• The length of the string can be anything (including zero length, if the command line argument is not provided).
	
	Let p(c = 'a') = m. You are given the following prior for m:
	• p(m = 0.1) = 0.9
	• p(m = 0.3) = 0.04
	• p(m = 0.5) = 0.03
	• p(m = 0.7) = 0.02
	• p(m = 0.9) = 0.01
	
	Obviously, all other possible values of m have probability 0.
	At the end, your program needs to report the posterior distribution of m given the data, as well as the computed p(c = 'a') (based on the posterior distribution of m). 
	The program output should follow EXACTLY this format:
	p(m = 0.1 | data) = %.4f
	p(m = 0.3 | data) = %.4f
	p(m = 0.5 | data) = %.4f
	p(m = 0.7 | data) = %.4f
	p(m = 0.9 | data) = %.4f
	p(c = 'a' | data) = %.4f
	
	Your program output should consist of only those lines, nothing else. 
	
  ### Running The Program 
  
   Execution commands:
   
       -python bayesian_estimate1.py
