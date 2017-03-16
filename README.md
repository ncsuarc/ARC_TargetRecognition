# Target Recognition

## Usage

### Step 1: Generate Training and Testing Data Set.

```bash
git clone https://github.com/ncsuarc/generate_targets.git
cd samples
python3 /PATH_TO/generate_targets/generate.py --number NUM
cd ../test_samples
python3 /PATH_TO/generate_targets/generate.py --number NUM
cd ..
```

### Step 2: Create Model
This is one example usage to create a relatively small model.
See the help for more information about how to use the parameters for creating models with different structures.
```bash
python3 create_model.py -s training -c 36 -f 16 -1 32 -1
```

### Step 3: Train

```bash
python3 train.py
```

### Step 4: Test

```bash
python3 test.py
```
