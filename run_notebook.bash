jupyter notebook --allow-root --no-browser \
     --ip=0.0.0.0 --port=8888 \
     --NotebookApp.token='' --NotebookApp.password=''\
     --MultiKernelManager.default_kernel_name='seesaw'\
     --KernelSpecManager.whitelist 'seesaw' \
     --NotebookApp.notebook_dir='/workdir/notebooks'\
     --KernelSpecManager.ensure_native_kernel=False