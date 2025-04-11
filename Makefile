install:
	pip install -r requirements.txt

train:
	jupyter nbconvert --to notebook --execute notebooks/ML.ipynb

forecast:
	python Forecast.py

clean:
	del /Q data\04_visualisation\*.csv
	del /Q data\04_visualisation\*.png
