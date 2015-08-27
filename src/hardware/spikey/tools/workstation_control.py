import pylogging as pylog
myLogger = pylog.get("PyN.wks")

import os
import xmltodict
import curses
import time
import numpy

if os.environ.has_key('PYNN_HW_PATH'):
    basePath = os.path.join(os.environ['PYNN_HW_PATH'], 'config')
else:
    raise EnvironmentError(
        'ERROR: The environment variable PYNN_HW_PATH is not defined!')
homePath = os.environ['HOME']


def getNextKey():
    '''get an unbuffered single character.'''

    import termios
    import fcntl
    import sys
    import os
    import time
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

    try:
        try:
            c = sys.stdin.read(1)
            return c
        except IOError:
            pass
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
    return None


def checkUserIsOwner(stationDict):
    '''Checks if the current user owns the workstation. If not, a warning is displayed once - giving a chance to cancel the action. Returns True if the process should continue, False otherwise.'''

    owner = stationDict['owner']
    import os
    import pwd
    user = pwd.getpwuid(os.geteuid())[0]

    if user == owner:
        return True

    print '\n'
    print '* * * * * * ! ! ! ATTENTION ! ! ! * * * * * *'
    print '*                                           *'
    print '* You are logged in as >>> ' + user + ' <<<.'
    print '* Current station belongs to >>> ' + owner + ' <<<.'
    print '*'
    print '* Press <space> to continue anyway or'
    print '*         <q>   to abort.'
    print '*'
    print '* Continuing automatically after 5 seconds.'
    print '*'
    print '* * * * * * * * * *'
    print '\n'

    import time
    try:
        for t in range(50):
            time.sleep(0.1)
            char = getNextKey()
            if char is None:
                pass
            elif char == 'q':
                print 'Canceled by user!'
                return False
            elif char == ' ':
                print "Continuing due to user's decision!"
                return True
            else:
                pass

        print "Continuing automatically after timeout!"
        return True

    except:
        print "Canceled due to error!"
        return False


def getWorkstation(workStationName=None, isFirstSetup=True):
    '''Returns a dictionary defining the user's workstation settings, provided that the files workstations.xml and my_stage1_station are set up correctly.'''

    global stationDict
    # check if work station configuration file is present and read
    # hardwareParameters
    if 'workstations.xml' in os.listdir(basePath):
        # check if user workstation definition file is present and check for
        # correct syntax
        if workStationName and workStationName != '':
            myStationString = workStationName
        elif 'my_stage1_station' in os.listdir(homePath):
            syntaxString = """ERROR: File my_stage1_station has to contain a string 'station<n>' (with <n> being an integer)!"""
            myStationFile = open(homePath + '/my_stage1_station', 'r')
            myStationString = myStationFile.read()
            myStationFile.close()
        else:
            raise Exception('ERROR: Could not find ' + homePath +
                            '/my_stage1_station! This file is necessary to determine your work station!')
        if myStationString != "nostation":
            if len(myStationString) < 8:
                raise Exception(syntaxString)
            if myStationString[0:7] != 'station':
                raise Exception(syntaxString)
            try:
                someNumber = int(myStationString[7:])
            except ValueError:
                raise Exception(syntaxString)
            formatOK = False
            # catch the case of a file with a newline or a space at the end
            while not formatOK:
                try:
                    # last symbol has to be an integer
                    someNumber = int(myStationString[-1])
                    formatOK = True
                except ValueError:
                    if len(myStationString) < 8:
                        raise Exception(syntaxString)
                    myStationString = myStationString[:-1]
            myLogger.info('Using station ' + myStationString)

        try:
            stationFile = open(basePath + '/workstations.xml', 'r')
            stationDict = xmltodict.xmltostringdict(stationFile)
            stationFile.close()
            if stationDict['availableStations'][myStationString]:
                myStationDict = stationDict[
                    'availableStations'][myStationString]
            else:
                raise Exception('ERROR: File ' + basePath + '/workstations.xml is present, but the key ' +
                                myStationString + ' defined in my_stage1_station can not be found.')

        except Exception, inst:
            raise Exception('ERROR: An error occured while trying to read work station data from file ' +
                            basePath + '/workstations.xml: ' + inst.__str__())

    else:
        raise Exception('ERROR: Could not find ' + basePath +
                        '/workstations.xml! This file is necessary to determine work station settings!')

#    if isFirstSetup and myStationString != "nostation":
#        if not checkUserIsOwner( myStationDict ): raise EnvironmentError("ERROR: Workstation doesn't belong to user.")

    try:
        voutCalibFile = os.environ['SPIKEYHALPATH'] + '/spikeycalib.xml'
        voutCalibHandler = open(voutCalibFile, 'r')
        voutDict = xmltodict.xml2obj(voutCalibHandler)
        voutCalibHandler.close()
        voutCalibDataAvailabe = False
        for s in voutDict.spikey:
            if int(s.nr) == int(myStationDict['spikeyNr']):
                # print 'found vout calib entry for spikey',int(s.nr)
                voutCalibDataAvailabe = True
                vmin = numpy.array(s.validMin.split(';'))
                vmin = numpy.array(vmin[:-1], float)
                vmax = numpy.array(s.validMax.split(';'))
                vmax = numpy.array(vmax[:-1], float)

                myStationDict['voutMins'] = vmin
                myStationDict['voutMaxs'] = vmax

                # for scope y-range calibration:
                # get maximum of v_rest, e_rev_I and v_reset values for the whole chip
                # vout indices of these voltages on left block are are v_rest:
                # 0, 1, e_rev_I: 2, 3, v_reset: 4, 5
                vlist = [0, 1, 2, 3, 4, 5]
                lowerVoltages = []
                for v in vlist:
                    for block in [0, 1]:
                        lowerVoltages.append(vmin[block * 25 + v])
                myStationDict['voutLower'] = max(lowerVoltages)
                if myStationDict['voutLower'] < 0.5:
                    # according to JS, v_rest can not be set to values < 0.5 V
                    # on Spikey
                    myStationDict['voutLower'] = 0.5
                # 1.1 V is a strict upper limit for possible membrane voltages,
                # still in some cases even the active threshold can be exceeded
                myStationDict['voutUpper'] = 1.2

        if not voutCalibDataAvailabe:
            myLogger.warn('For Spikey ' + str(int(myStationDict[
                          'spikeyNr'])) + ' no vout ranges were found in file ' + os.environ['SPIKEYHALPATH'] + '/spikeycalib.xml')
            myLogger.info('Using default range [0.2 V, 1.6 V]')
            myStationDict['voutLower'] = 0.5
            myStationDict['voutUpper'] = 1.2
            myStationDict['voutMins'] = numpy.ones(50) * 0.2
            myStationDict['voutMaxs'] = numpy.ones(50) * 1.6

    except Exception, inst:
        raise Exception('ERROR: A problem occured while trying to read data from file ' +
                        os.environ['SPIKEYHALPATH'] + '/spikeycalib.xml: ' + inst.__str__())

    return myStationDict


def getDefaultWorkstationName():
    '''Returns the default workstation name found in the file my_stage1_station.'''

    global stationDict
    # check if work station configuration file is present and read
    # hardwareParameters
    if 'my_stage1_station' in os.listdir(homePath):
        myStationFile = open(homePath + '/my_stage1_station', 'r')
        myStationString = myStationFile.read()
        myStationFile.close()
    else:
        raise Exception("ERROR: No my_stage1_station file found!")
    return myStationString
