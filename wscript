#!/usr/bin/env python
import sys
from waflib import Options
APPNAME='pyNN.hardware.spikey'


def depends(ctx):
    ctx('logger', 'pylogging')
    ctx('spikeyhal', branch='flyspi')

def options(opt):
    pass

def configure(conf):
    pass

spikey_env_template = """\
export PYTHONPATH=$PYTHONPATH:{0}/lib:{0}/lib/python{1}/site-packages
export PYNN_HW_PATH={0}/deb-pynn/src/hardware/spikey
export SPIKEYHALPATH={0}/spikeyhal
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{0}/lib
export PATH=$PATH:{0}/bin"""

def build(bld):
    this_dir = bld.path.get_src().abspath()
    # FIXME: ask distutils/setup.py for relative site-package path (to prefix)
    bld.exec_command('python setup.py install --prefix={}'.format(bld.env.PREFIX), cwd=this_dir)

    ver = '.'.join([str(x) for x in sys.version_info[:2]])
    bld(
        target = 'env.sh',
        rule = 'echo \'{0}\' > '.format(spikey_env_template.format(bld.env.PREFIX, ver)) + '${TGT}',
        shell = True,
        install_path = 'bin'
    )
