if [ -z $VIRTUAL_ENV ]; then
    python_version=$(python3 --version | awk '{print $2}' | awk -F. '{printf "%s.%s\n",$1,$2}')
    echo "Enable virtual env with python version $python_version"
    virtualenv .venv
    . .venv/bin/activate
    
    if [ -f requirements.txt ]; then
	pip install -r requirements.txt --target $VIRTUAL_ENV/lib/python${python_version}/site-packages
	#pip install -r requirements.txt
    fi
fi
#export PYTHONPATH=$VIRTUAL_ENV/lib/python${python_version}/site-packages
function venv_exit() {
    deactivate
    unset PYTHONPATH
}
