with  open('EarIntervalsNegative10.txt', 'a') as outFile:

    inizio = 800
    fine = 1000

    while inizio < fine: 
        outFile.write("%s %s" % (inizio, inizio+12))
        outFile.write("\n")
        inizio += 18
