from seesaw.memory_cache import ReferenceCache
import ray
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='control the data cache. calling it again will restart the cache')
    #parser.add_argument('--restart', type=int, action='store_true',  help='restart the cache')
    args = parser.parse_args()

    ray.init('auto', namespace='seesaw')

    actor_name = 'actor#cache'
    try:
      oldh = ray.get_actor(actor_name)
      print('found old cache actor, destroying it')
      ray.kill(oldh)
      print('ended previous cache actor')
    except:
      pass
      # no actor to kill

    print('starting new cache actor')
    h = ray.remote(ReferenceCache).options(name=actor_name, num_cpus=1, lifetime='detached').remote()
    r = h.ready.remote()
    ray.get(r)
    print('new cache actor ready')