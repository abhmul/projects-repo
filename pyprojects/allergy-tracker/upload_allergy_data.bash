#!/bin/bash
cd $DRIVE_PATH
drive pull --no-prompt sync-documents/

cd $PROJECTS/projectsrepo/allergy-tracker && {
	eval "$(conda shell.bash hook)" && \
	conda activate projects && \
	coconut --target=3.7 data_to_csv.coco && \
	python data_to_csv.py 95445 --allergy_csv $SYNC_DOCUMENTS/data/allergy_index.csv --asthma_csv $SYNC_DOCUMENTS/data/asthma_index.csv --disease_csv $SYNC_DOCUMENTS/data/disease_index.csv && \
	cd $DRIVE_PATH && \
	drive push --no-prompt sync-documents/
}
