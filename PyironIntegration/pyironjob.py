from pyiron_base.utils.error import ImportAlarm
from pyiron_base.jobs.job.template import TemplateJob

try:
    from .RobinRollandModel.new_main import RRModel
except ImportError:
    import_alarm = ImportAlarm("Unable to import RobinRollandmodel")

RRmodel()