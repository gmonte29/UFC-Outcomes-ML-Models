make install:
	python3 -m venv UFC_ENV && \
	source UFC_ENV/bin/activate && \
	pip install --upgrade pip && \
	pip install pandas scikit-learn tensorflow matplotlib kaggle && \
	kaggle datasets download -d rajeevw/ufcdata && \
	unzip ufcdata.zip && \
	rm ufcdata.zip && \
	rm data.csv && \
	rm raw_fighter_details.csv && \
	rm raw_total_fight_data.csv

make clean:
	rm preprocessed_data.csv
	conda deactivate
	rm -r UFC_ENV
	rm -r logs
