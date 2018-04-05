ag --tex -l | entr -s 'echo `date +%s` `detex thesis.tex| wc -w` >> ~/Sync/log/pdf/pdflog.txt'

