python3 -m pip install tensorflow[and-cuda]

dataset: https://www.jottacloud.com/s/16192c1b7d186444bd1b7e5c16292e42e9e/thumbs

ls |grep oil

MLnode kjøring:  module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

fox: module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0


pip install --user matplotlib==3.7.3

ubuntu: 

pwd    path til current dir

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

kopiere filer fra ml nodes til min pc: scp -J joakiast@login.uio.no joakiast@ml9.hpc.uio.no:archive_name.tar.gz  .

tar file: tar -czf archive_name.tar.gz generated_images/

ssh ec-joakims@fox.educloud.no     
brukernavnet er ec-joakims

overføre filer til fox
scp test1.tar.gz ec-joakims@fox.educloud.no:


overføre filer fra fox sender filer til og fra docs: scp ec-joakims@fox.educloud.no:~/Documents/simulated_data_image_translation.tar.gz ~/Documents/


fox info: https://documentation.sigma2.no/training/events/2023-04-hpc-on-boarding.html

#=============================================== i fox 


module load Python/3.10.4-GCCcore-11.3.0

source tensor_env/bin/activate

module load CUDA/12.2.2

module load cuDNN/8.4.1.50-CUDA-11.7.0







se på for cycleGAN: https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

instance norm: 
https://arxiv.org/pdf/1607.08022.pdf
