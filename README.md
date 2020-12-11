# Sp17_bcs_comsats-Project



To install all the dependencies:
	`pip install -r requirements.txt`
	`$ cd tiny_yolo/weights/`
    	`$ bash download_weights.sh`
To run:
python main.py 

## Command line arguments:

-a: 
Diplaying Momentary Time to Contact without Acceleration (point 2 of research paper)
 _Otherwise it will display with modeling of acceleration which is giving negative results (point 3 of research paper)_
 
 python main.py -a


-c:
To use camera

python main.py -c


-r:
To use time of 60 fps




### Combinations can also be used:
To display TTC without acceleration using camera

 _Sequence doesn't matter_
 
python main.py -c -a
