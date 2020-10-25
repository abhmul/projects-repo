#!/bin/bash
cd $DRIVE_PATH
drive pull --no-prompt sync-documents/

cd $PROJECTS/projects/allergy-tracker && {
	eval "$(conda shell.bash hook)" && \
	conda activate projects && \
	coconut --target=3.7 data_to_csv.coco && \
	python data_to_csv.py 45458 -s 2020-09-21 --allergy_csv $SYNC_DOCUMENTS/data/allergy_index.csv --asthma_csv $SYNC_DOCUMENTS/data/asthma_index.csv --disease_csv $SYNC_DOCUMENTS/data/disease_index.csv && \
	cd $DRIVE_PATH && \
	drive push --no-prompt sync-documents/
}
