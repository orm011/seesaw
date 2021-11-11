from asyncio import Event
from typing import Tuple
from time import sleep
import ray

from ray.util import ActorPool

# For typing purposes
from ray.actor import ActorHandle
from tqdm.auto import tqdm
import copy

## Actor pool interface is different. 
# no need for the progress bar actor there...
## create a new pool every time because 
# in case of interruption, actor pool state seems
def tqdm_map(actors, actor_tup_function, tups, res=None):
    assert res is not None, 'provide way to save partial results'
    
    initial_len = len(res)
    actor_pool = ActorPool(actors)
    for tup in tups:
        actor_pool.submit(actor_tup_function, tup)

    pbar = tqdm(total=len(tups))
    while True:
        nxt = actor_pool.get_next_unordered()
        ## copy to free up any references at the source
        res.append(copy.deepcopy(nxt))
        pbar.update(1)
        if (len(res) - initial_len) == len(tups):
            print('done with new tups')
            break
            
    pbar.close()
    return res

# taken from https://docs.ray.io/en/master/auto_examples/progress_bar.html
# use with pool:
# pbar = ProgressBar(total)
# pexps = pool.map_async(pbar.wrap(mapfun), tups)
# pbar.print_until_done()

@ray.remote(num_cpus=.01)
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter

class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

    def wrap(self, fun):
        pbar_actor = self.actor
        def wrapped(*args, **kwargs):
            try:
                res = fun(*args, **kwargs)
                return res
            finally:
                pbar_actor.update.remote(1)
            
        return wrapped