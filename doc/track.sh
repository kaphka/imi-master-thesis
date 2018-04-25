ag -g pdf | entr -s 'echo `date +%s` `pdftotext thesis.pdf - | wc -w` >> ~/Sync/log/pdf/pdflog.txt'

