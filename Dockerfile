FROM anibali/pytorch:cuda-8.0

# install jupyter
RUN conda install -y jupyter 

# get additional requirements - conda
RUN conda install -y tqdm
RUN conda install -y pandas

RUN conda install -c -y spacy \
	&& python -m spacy download en \
	&& python -m spacy download de

RUN conda install -y PrettyTable
RUN conda install -c -y beautifulsoup4
RUN conda install -y hyperopt
RUN conda install -y scikit-learn \
	&& conda clean -ya

# get additional requirements - pip
RUN pip install torchtext
RUN pip install stop-words
RUN pip install pyspellchecker
RUN pip install revtok
RUN pip install ElementTree
RUN pip install tensorboaordX

CMD ["jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]

