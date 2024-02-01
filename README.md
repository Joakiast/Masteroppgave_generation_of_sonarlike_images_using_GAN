python3 -m pip install tensorflow[and-cuda]

dataset: https://www.jottacloud.com/s/16192c1b7d186444bd1b7e5c16292e42e9e/thumbs

ls |grep oil

MLnode kjøring:  module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

pip install --user matplotlib==3.7.3

ubuntu: 

pww    path til current dir

ls |grep oil   lister ut alle objeter som heter noe med oil

ls -1 | wc -l  lister ut antall objeker i current dir

scp -J joakiast@login.uio.no -r train/ joakiast@ml9.hpc.uio.no:

login til mlnode ssh -J joakiast@gothmog.uio.no  joakiast@ml9.hpc.uio.no

tmux ubuntu mlnode brukes for å kunne kjøre trening selvom pc er av, fremgangsmåte: 

tmux new -s [navn på session] 
ctrl+B D 
Nå er det trygt å logge av, prosessen kjører. 

for å komme til prosessen igjen: 
tmux attach -t [navn på session]

kopiere filer fra ml nodes til min pc: scp -J joakiast@login.uio.no joakiast@ml9.hpc.uio.no:archive.tar.gz  .

