# from .runner import run_net
from .runner import test_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune_cls import run_net as finetune_run_net_cls
from .runner_finetune_cls import test_net as test_run_net

from .runner_finetune_partseg import run_net as finetune_run_net_partseg
from .runner_finetune_dg import run_net as finetune_run_net_dg
