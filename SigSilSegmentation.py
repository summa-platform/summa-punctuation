#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Alexandre Nanchen"
__version__     = "Revision: 1.0"
__date__        = "Date: 2018/05"
__copyright__   = "Copyright (c) 2018 Idiap Research Institute"
__license__     = "See COPYING"

import logging

from statsmodels.distributions.empirical_distribution import ECDF

class SigSilSegmentation:
    """Segmentation using "significant silence" criterion.
    """
    logger              = logging.getLogger("summa.sig-sil-seg")

    NBINS               = 40
    WORDINDICE          = 0
    PAUSEDURATIONINDICE = 3

    def __init__(self, cdfThreshold=0.97):
        """ Default constructor.
        """
        self.cdfThreshold = cdfThreshold

    ########################
    # Public implementation
    #
    def significantSilenceSegmentation(self, alignmentList):
        """Segmentation of a words sequence using duration only.

           alignmentList is assumed to be:
           [{"word": strWord, "time": startTime, "duration": duration}, ...]

           return a sentences list
        """
        featsList = self._computePauseDurations(alignmentList)

        #Words sequence
        wordsList, pauseDurationList = [], []
        for i in range(len(featsList)):
            wordsList.append(featsList[i][self.WORDINDICE])
            pauseDurationList.append(featsList[i][self.PAUSEDURATIONINDICE])

        #Segmentation
        sentencesList = self._segmentWithSignificantSilence(wordsList, pauseDurationList,
                                                            self.cdfThreshold)
        return sentencesList


    def significantSilenceSegmentationFile(self, alignmentFile):
        """Read an alignment file and perform segmentation.
        """
        contentList = []
        with open(alignmentFile, 'r') as f:
            line = f.readline()
            while line != "":
                lineList = line.split(' ')
                contentList.append({'word': lineList[3],
                                    'time': float(lineList[1]),
                                    'duration': float(lineList[2])})
                line = f.readline()

        return self.significantSilenceSegmentation(contentList)

    ########################
    # Implementation
    #
    @staticmethod
    def _segmentWithSignificantSilence(wordsList, pauseDurationList, threshold):
        """Build an ECDF from 'pauseDurationList' and threshold at 'threshold'
           value.

           return a sentences list
        """
        ecdf = ECDF(pauseDurationList)

        sentStartIndice, sentencesList = 0, []

        for i, sil in enumerate(pauseDurationList):
            cdfValue = ecdf(sil)
            #print cdfValue
            if cdfValue > threshold:
                strSentence = u" ".join(wordsList[sentStartIndice:i+1])
                sentencesList.append(strSentence)
                sentStartIndice = i+1

        if sentStartIndice < len(pauseDurationList):
            strSentence = u" ".join(wordsList[sentStartIndice:len(pauseDurationList)])
            sentencesList.append(strSentence)

        return sentencesList

    @staticmethod
    def _computePauseDurations(wordsList):
        """Extract pause duration from a words list.

           wordsList is assume to be:
            [{"word": strWord, "time": startTime, "duration": duration}, ...]
        """
        ctmList = SigSilSegmentation._getCTMList(wordsList)

        #Pause duration computation
        outputList, maxSilence = [], 0

        for i, l in enumerate(ctmList):
            #Default to end of sentence
            silenceDuration = None

            #Not end word
            if i+1 < len(ctmList):
                #Silence duration
                endTime = float(ctmList[i][1]) + float(ctmList[i][2])
                nextStartTime = float(ctmList[i+1][1])
                silenceDuration = nextStartTime - endTime

                if silenceDuration < 0:
                    silenceDuration = 0

                if silenceDuration > maxSilence:
                    maxSilence = silenceDuration

            l.append(silenceDuration)
            outputList.append(l)

        #End of sentences and normalization
        for i, l in enumerate(outputList):
            if outputList[i][3] == None:
                outputList[i][3] = maxSilence
            outputList[i][3] = round(outputList[i][3], 6)

        return outputList

    @staticmethod
    def _getCTMList(wordsList):
        """wordsList is assume to be:
            [{"word": strWord, "time": startTime, "duration": duration}, ...]
        """
        #CTM list generation
        ctmList = []
        for segment in wordsList:
            strWord = segment["word"]
            startTime = float(segment["time"])
            duration = float(segment["duration"])

            ctmList.append([strWord, startTime, duration])

        return ctmList
