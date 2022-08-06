challenge-template:
	touch template.tar.gz
	tar --exclude=.git --exclude=.gitignore --exclude=template.tar.gz \
	  --exclude=*.csv.gz --exclude=models* --exclude=__pycache__* \
	  --exclude=.vscode* --exclude=*.ipynb --exclude=data* --exclude=env \
	  --exclude=scoring_function.tar.gz --exclude=Makefile \
	  --exclude=.gitlab-ci.yml \
	  -cvzf template.tar.gz .

scoring-function:
	tar cvzf scoring_function.tar.gz score.py
