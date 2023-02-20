{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 246,
   "id": "f100648f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import GPy\n",
    "import emukit.multi_fidelity\n",
    "import emukit.test_functions\n",
    "from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper\n",
    "from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import colors as mcolors\n",
    "colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)\n",
    "%matplotlib inline\n",
    "\n",
    "## Generate samples from the Forrester function\n",
    "\n",
    "\n",
    "\n",
    "x_train_l =  np.array([[0.5,0.42],[0.50,0.65],[0.50,1],[0.5,1.35],[0.5,1.58],[0.7,0.42],[0.7,0.65],[0.7,1],[0.7,1.35],[0.7,1.58],[1,0.42],[1,0.65],[1,1],[1,1.35],[1,1.58],\\\n",
    "[1.3,0.42],[1.3,0.65],[1.3,1],[1.3,1.35],[1.3,1.58],[1.5,0.42],[1.5,0.65],[1.5,1],[1.5,1.35],[1.5,1.58]])\n",
    "x_train_h = np.array([[0.5,1],[1,0.42],[1,1],[1,1.58],[1.5,1]])\n",
    "y_train_l = np.array([0.4986808747,\n",
    "0.475976665,\n",
    "0.4777031069,\n",
    "0.4923418646,\n",
    "0.5028265703,\n",
    "0.4753060066,\n",
    "0.4595472371,\n",
    "0.4620449908,\n",
    "0.4691087625,\n",
    "0.4742488188,\n",
    "0.4513367251,\n",
    "0.4396361379,\n",
    "0.4368311097,\n",
    "0.437009123,\n",
    "0.4393375827,\n",
    "0.4180588293,\n",
    "0.4084524096,\n",
    "0.404768885,\n",
    "0.4059021486,\n",
    "0.40739032015181575,\n",
    "0.3861760934,\n",
    "0.3767324971,\n",
    "0.3724887529,\n",
    "0.3747228606,\n",
    "0.3767415899])\n",
    "y_train_h = np.array([\n",
    "0.5222562742,\n",
    "\n",
    "0.4187630715,\n",
    "0.3821620197,\n",
    "0.4202147777,\n",
    "\n",
    "0.3428283998159303\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "id": "dab3bb7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "flag_mf = []\n",
    "for i in range(len(X_train)):\n",
    "    if i>len(x_train_l):\n",
    "       flag_mf.append(1)\n",
    "    else:\n",
    "        flag_mf.append(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 248,
   "id": "4a232731",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]\n"
     ]
    }
   ],
   "source": [
    "print(flag_mf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 249,
   "id": "283df994",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_l=np.c_[ x_train_l, np.zeros(len(x_train_l)) ]   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "id": "34b36efc",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_h=np.c_[ x_train_h, np.ones(len(x_train_h)) ]   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 251,
   "id": "fce65d63",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.5  0.42 0.  ]\n",
      " [0.5  0.65 0.  ]\n",
      " [0.5  1.   0.  ]\n",
      " [0.5  1.35 0.  ]\n",
      " [0.5  1.58 0.  ]\n",
      " [0.7  0.42 0.  ]\n",
      " [0.7  0.65 0.  ]\n",
      " [0.7  1.   0.  ]\n",
      " [0.7  1.35 0.  ]\n",
      " [0.7  1.58 0.  ]\n",
      " [1.   0.42 0.  ]\n",
      " [1.   0.65 0.  ]\n",
      " [1.   1.   0.  ]\n",
      " [1.   1.35 0.  ]\n",
      " [1.   1.58 0.  ]\n",
      " [1.3  0.42 0.  ]\n",
      " [1.3  0.65 0.  ]\n",
      " [1.3  1.   0.  ]\n",
      " [1.3  1.35 0.  ]\n",
      " [1.3  1.58 0.  ]\n",
      " [1.5  0.42 0.  ]\n",
      " [1.5  0.65 0.  ]\n",
      " [1.5  1.   0.  ]\n",
      " [1.5  1.35 0.  ]\n",
      " [1.5  1.58 0.  ]\n",
      " [0.5  1.   1.  ]\n",
      " [1.   0.42 1.  ]\n",
      " [1.   1.   1.  ]\n",
      " [1.   1.58 1.  ]\n",
      " [1.5  1.   1.  ]]\n",
      "[0.49868087 0.47597667 0.47770311 0.49234186 0.50282657 0.47530601\n",
      " 0.45954724 0.46204499 0.46910876 0.47424882 0.45133673 0.43963614\n",
      " 0.43683111 0.43700912 0.43933758 0.41805883 0.40845241 0.40476889\n",
      " 0.40590215 0.40739032 0.38617609 0.3767325  0.37248875 0.37472286\n",
      " 0.37674159 0.52225627 0.41876307 0.38216202 0.42021478 0.3428284 ]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "X_train=np.append(x_train_l,x_train_h,axis=0)\n",
    "Y_train=np.append(y_train_l,y_train_h,axis=0)\n",
    "print(X_train)\n",
    "print(Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "id": "a55545a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n",
      "[[0.49868087]\n",
      " [0.47597667]\n",
      " [0.47770311]\n",
      " [0.49234186]\n",
      " [0.50282657]\n",
      " [0.47530601]\n",
      " [0.45954724]\n",
      " [0.46204499]\n",
      " [0.46910876]\n",
      " [0.47424882]\n",
      " [0.45133673]\n",
      " [0.43963614]\n",
      " [0.43683111]\n",
      " [0.43700912]\n",
      " [0.43933758]\n",
      " [0.41805883]\n",
      " [0.40845241]\n",
      " [0.40476889]\n",
      " [0.40590215]\n",
      " [0.40739032]\n",
      " [0.38617609]\n",
      " [0.3767325 ]\n",
      " [0.37248875]\n",
      " [0.37472286]\n",
      " [0.37674159]\n",
      " [0.52225627]\n",
      " [0.41876307]\n",
      " [0.38216202]\n",
      " [0.42021478]\n",
      " [0.3428284 ]]\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape[1])\n",
    "Y_train=np.reshape(Y_train,(len(X_train),1))\n",
    "print(Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "id": "926d9628",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization restart 1/50, f = 104252.31435851895\n",
      "Optimization restart 2/50, f = 104254.20842972239\n",
      "Optimization restart 3/50, f = 104247.99289935426\n",
      "Optimization restart 4/50, f = 104250.47275054509\n",
      "Optimization restart 5/50, f = 104251.01493440388\n",
      "Optimization restart 6/50, f = 104249.68196904246\n",
      "Optimization restart 7/50, f = 104262.13725057486\n",
      "Optimization restart 8/50, f = 104250.51605425186\n",
      "Optimization restart 9/50, f = 104274.03839049142\n",
      "Optimization restart 10/50, f = 104256.59766025619\n",
      "Optimization restart 11/50, f = 104253.32399996747\n",
      "Optimization restart 12/50, f = 104247.27617135839\n",
      "Optimization restart 13/50, f = 104254.80742409051\n",
      "Optimization restart 14/50, f = 104251.85724517316\n",
      "Optimization restart 15/50, f = 104249.33677793456\n",
      "Optimization restart 16/50, f = 104254.82560440328\n",
      "Optimization restart 17/50, f = 104261.86814499662\n",
      "Optimization restart 18/50, f = 104260.24413937033\n",
      "Optimization restart 19/50, f = 104257.65978093904\n",
      "Optimization restart 20/50, f = 104251.67615602376\n",
      "Optimization restart 21/50, f = 104247.27147871107\n",
      "Optimization restart 22/50, f = 104262.19552687934\n",
      "Optimization restart 23/50, f = 104247.58858982733\n",
      "Optimization restart 24/50, f = 104249.66433536467\n",
      "Optimization restart 25/50, f = 104255.22943272258\n",
      "Optimization restart 26/50, f = 104256.7995667541\n",
      "Optimization restart 27/50, f = 104253.8333023515\n",
      "Optimization restart 28/50, f = 104256.69010315607\n",
      "Optimization restart 29/50, f = 104261.020593941\n",
      "Optimization restart 30/50, f = 104263.95419682289\n",
      "Optimization restart 31/50, f = 104255.6619220338\n",
      "Optimization restart 32/50, f = 104247.69820988191\n",
      "Optimization restart 33/50, f = 104260.1452559585\n",
      "Optimization restart 34/50, f = 104260.92802326204\n",
      "Optimization restart 35/50, f = 104249.56544145565\n",
      "Optimization restart 36/50, f = 104256.60034853117\n",
      "Optimization restart 37/50, f = 104259.7887416776\n",
      "Optimization restart 38/50, f = 104260.47397324425\n",
      "Optimization restart 39/50, f = 104253.17359666174\n",
      "Optimization restart 40/50, f = 104261.46190598553\n",
      "Optimization restart 41/50, f = 104255.27305306685\n",
      "Optimization restart 42/50, f = 104253.66422085158\n",
      "Optimization restart 43/50, f = 104258.65206071764\n",
      "Optimization restart 44/50, f = 104259.66766168727\n",
      "Optimization restart 45/50, f = 104250.2382998425\n",
      "Optimization restart 46/50, f = 104254.78786208107\n",
      "Optimization restart 47/50, f = 104251.84319288415\n",
      "Optimization restart 48/50, f = 104252.8500477555\n",
      "Optimization restart 49/50, f = 104252.37857574043\n",
      "Optimization restart 50/50, f = 104261.06819830171\n"
     ]
    }
   ],
   "source": [
    "kernels = [GPy.kern.RBF(1),GPy.kern.RBF(1)]\n",
    "lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)\n",
    "gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)\n",
    "gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)\n",
    "gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)\n",
    "\n",
    "lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=50)\n",
    "\n",
    "## Fit the model\n",
    "  \n",
    "lin_mf_model.optimize()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "id": "d06a40a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Compute mean and variance predictions\n",
    "testx = np.linspace(0.5,1.5, 101, endpoint=True)\n",
    "\n",
    "testy= np.linspace(0.42,1.58, 101, endpoint=True)\n",
    "\n",
    "\n",
    "x = [(a, b) for a in testx for b in testy] \n",
    "x=np.array([x])\n",
    "x_plot=np.reshape(x,[10201,2])\n",
    "X_plot_l=np.c_[ x_plot, np.zeros(len(x_plot)) ]   \n",
    "X_plot_h=np.c_[ x_plot, np.ones(len(x_plot)) ]   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 255,
   "id": "db623716",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "## Compute mean predictions and associated variance\n",
    "\n",
    "## Compute mean predictions and associated variance\n",
    "\n",
    "lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(X_plot_l)\n",
    "\n",
    "hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(X_plot_h)\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "id": "21c16ea4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.48950584]\n",
      " [0.48950584]\n",
      " [0.48950584]\n",
      " ...\n",
      " [0.37737262]\n",
      " [0.37737262]\n",
      " [0.37737262]]\n"
     ]
    }
   ],
   "source": [
    "print(lf_mean_lin_mf_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "id": "ee2507b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.52225626]\n",
      " [0.52225626]\n",
      " [0.52225626]\n",
      " ...\n",
      " [0.34282839]\n",
      " [0.34282839]\n",
      " [0.34282839]]\n"
     ]
    }
   ],
   "source": [
    "print(hf_mean_lin_mf_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 258,
   "id": "20a38513",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Create standard GP model using only high-fidelity data\n",
    "x_train_DNS=np.array([[0.5,0.42],[0.5,1],[0.5,1.58],[1,0.42],[1,1],[1,1.58],[1.5,0.42],[1.5,1],[1.5,1.58]])\n",
    "y_train_DNS=np.array([0.5047597937,\n",
    "0.5222562742,\n",
    "0.5985461618,\n",
    "0.4187630715,\n",
    "0.3821620197,\n",
    "0.4202147777,\n",
    "0.2783293064,\n",
    "0.3080605842,\n",
    "0.3428283998159303\n",
    "])\n",
    "x_train_DNS=np.c_[ x_train_DNS, np.ones(len(x_train_DNS)) ]   \n",
    "y_train_DNS=np.reshape(y_train_DNS,(len(x_train_DNS),1))\n",
    "\n",
    "\n",
    "kernel = GPy.kern.RBF(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 259,
   "id": "94a2bb92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization restart 1/5, f = 399363.7885887812\n",
      "Optimization restart 2/5, f = 399363.794431367\n",
      "Optimization restart 3/5, f = 399363.786868795\n",
      "Optimization restart 4/5, f = 399363.786786003\n",
      "Optimization restart 5/5, f = 399363.7869613745\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " /home/mech/smdsouza/.local/lib/python3.9/site-packages/GPy/core/gp.py:85: UserWarning:Your kernel has a different input dimension 1 then the given X dimension 3. Be very sure this is what you want and you have not forgotten to set the right input dimenion in your kernel\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<paramz.optimization.optimization.opt_lbfgsb at 0x7f9386ca0880>,\n",
       " <paramz.optimization.optimization.opt_lbfgsb at 0x7f938c924b50>,\n",
       " <paramz.optimization.optimization.opt_lbfgsb at 0x7f9386e3eca0>,\n",
       " <paramz.optimization.optimization.opt_lbfgsb at 0x7f9386da1be0>,\n",
       " <paramz.optimization.optimization.opt_lbfgsb at 0x7f9386dcdf70>]"
      ]
     },
     "execution_count": 259,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "high_gp_model = GPy.models.GPRegression(x_train_DNS, y_train_DNS, kernel)\n",
    "high_gp_model.Gaussian_noise.fix(0)\n",
    "\n",
    "## Fit the GP model\n",
    "\n",
    "high_gp_model.optimize_restarts(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 260,
   "id": "43ccdd52",
   "metadata": {},
   "outputs": [],
   "source": [
    "hf_mean_high_gp_model, hf_var_high_gp_model  = high_gp_model.predict(x_plot)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 261,
   "id": "dc8490e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.54185397]\n",
      " [0.54185397]\n",
      " [0.54185397]\n",
      " ...\n",
      " [0.30973934]\n",
      " [0.30973934]\n",
      " [0.30973934]]\n"
     ]
    }
   ],
   "source": [
    "print(hf_mean_high_gp_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 262,
   "id": "77fd6ae5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA/UAAAPbCAYAAAAdHau/AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAA9hAAAPYQGoP6dpAACHG0lEQVR4nOzdd3iddf3/8VfaQotI4kCGMlU2olCQJQgUyhYUtIgWUYYooAgOEP3KUAEFWTJF9ipQkD3KEhAcIOBAEVEEsQVZCQh05fz+uH9pOtLkJE1yn3PyeFxXLnJGTt75XtdXefq578+nqVKpVAIAAADUnWFlDwAAAAD0jagHAACAOiXqAQAAoE6JegAAAKhToh4AAADqlKgHAACAOiXqAQAAoE6JegAAAKhTI8oeoBa1t7fnP//5TxZbbLE0NTWVPQ4AAAANrlKp5NVXX8273/3uDBtW/fq7qO/Cf/7znyy77LJljwEAAMAQ88wzz2SZZZap+v2ivguLLbZYkuL/mM3NzSVPAwAAQKNra2vLsssuO6tHqyXqu9BxyX1zc7OoBwAAYND09hZwG+UBAABAnRL1AAAAUKdEPQAAANQpUQ8AAAB1StQDAABAnSo96k8//fSsuOKKGTVqVEaPHp177713vu+9++6709TUNM/XX//611nvOf/887t8z5tvvjkYfw4AAAAMmlKPtJswYUIOOuignH766dl4441z1llnZdttt81jjz2W5ZZbbr4/9/jjj89x1Ny73vWuOV5vbm7O448/Psdzo0aN6t/hAQAAoGSlRv1PfvKT7LXXXtl7772TJCeddFJuvfXWnHHGGTnmmGPm+3NLLLFE3va2t8339aampiy11FL9PS4AAADUlNIuv582bVoeeuihjB07do7nx44dm/vvv7/bn1177bWz9NJLZ8yYMbnrrrvmef21117L8ssvn2WWWSY77LBDHn744W4/b+rUqWlra5vjCwAAAGpdaVH/wgsvZObMmVlyySXneH7JJZfMlClTuvyZpZdeOmeffXYmTpyYq6++OqusskrGjBmTe+65Z9Z7Vl111Zx//vm57rrrctlll2XUqFHZeOON88QTT8x3lmOOOSYtLS2zvpZddtn++SMBAABgADVVKpVKGb/4P//5T97znvfk/vvvz4Ybbjjr+R/84Ae56KKL5tj8rjs77rhjmpqact1113X5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R9ZVdV/9mWcmX/pS8f3XvpaccIKgB2gAfe3Q0i6/nzZtWh566KGMHTt2jufHjh2b+++/f74/d9555+XJJ5/M9773vfm+57XXXsvyyy+fZZZZJjvssEMefvjhbmeZOnVq2tra5vgCAKg5p5/eGfSHHCLoASgv6l944YXMnDkzSy655BzPL7nkkpkyZUqXP/PEE0/k0EMPzSWXXJIRI7q+c2DVVVfN+eefn+uuuy6XXXZZRo0alY033jhPPPHEfGc55phj0tLSMutr2WWX7fsfBgAwl2pW6Xt832mnFbvbJ8k3vpH8+MeCHoDyN8prmuu/jCqVyjzPJcnMmTOz++6758gjj8zKK68838/bYIMN8tnPfjYf/OAHs8kmm+SKK67IyiuvnFNPPXW+P3PYYYeltbV11tczzzzT9z8IAKC/nXpqcsABxfff/GZy3HGCHoAkJW6Ut/jii2f48OHzrMo///zz86zeJ8mrr76aBx98MA8//HAO+P//pdbe3p5KpZIRI0bktttuyxZbbDHPzw0bNizrrbdetyv1I0eOzMiRIxfwLwIAGACnnFLsbp8khx6a/PCHgh6AWUpbqV944YUzevToTJo0aY7nJ02alI022mie9zc3N+ePf/xjHnnkkVlf++23X1ZZZZU88sgjWX/99bv8PZVKJY888kiWXnrpAfk7AAB6UtUGeF2976STOoP+298W9ADMo9Qj7Q4++OCMHz8+6667bjbccMOcffbZefrpp7PffvslKS6Lf/bZZ3PhhRdm2LBhWXPNNef4+SWWWCKjRo2a4/kjjzwyG2ywQVZaaaW0tbXllFNOySOPPJLTTjttUP82AIAFcuKJycEHF98ffnhy9NGCHoB5lBr148aNy4svvpijjjoqkydPzpprrpmbbropyy+/fJJk8uTJefrpp3v1ma+88kr23XffTJkyJS0tLVl77bVzzz335MMf/vBA/AkAAFV56tjtu90Ib45V+hNOSL7+9eL77343OfJIQQ9Al0o9p75WOaceAOir2cN9fpfdd/ue448vdrdPku99LzniiP4eEYAa1NcOFfVdEPUAQG9VvQrfnR/9KPnWt4rvjziiiHoAhoS+dmjpR9oBANS7ns6hr+qc+mOP7Qz6I48U9ABURdQDAJTtmGOSww4rvj/qqOT//q/ceQCoG6IeAGABVLUK3937fvCD4ri6JPn+94uN8QCgSqIeAKAsRx+dfOc7xfc//GFxdB0A9IKoBwAow+yX2c9++T0A9IKoBwBYANXubD/H+2bf2f6445JDD+3/wQAYEkaUPQAAwJBRqRRBf9RRxeMf/zj5+tdLHQmA+ibqAQCqNPtmd7OvvD917PY9n1NfqRSr80cfXTx5/PHJIYcM2KwADA1NlUqlUvYQtaatrS0tLS1pbW1Nc3Nz2eMAACXrMdjn895Zr1Uqxa72P/hB8fgnP0m+9rV+nxOA+tXXDhX1XRD1AECHao6s6/a++kql2NX+mGOKxyeemBx0UP8MB0DD6GuHuvweAGCgVCrFGfTHHls8Pvnk5CtfKXcmABqK3e8BAOajmlX6+b6vUil2te8I+lNPFfQA9Dsr9QAA/a1SSb75zWIzvCT56U+T/fcvdyYAGpKoBwDoT5VK8o1vJCecUDw+7bTky18udyYAGpbL7wEA5qPbDfC6el+lUhxT1xH0Z5wh6AEYUFbqAQD6Q6VSHFN38snF47POSvbdt9yZAGh4VuoBALrR02r9U8duXwT9QQd1Bv3ZZwt6AAaFlXoAgMy5g/3cId/xuMv3VCrFrvY//WnS1JT87GfJXnsN/MAAkKSpUqlUyh6i1rS1taWlpSWtra1pbm4uexwAYAB1d2xdj/fUVyrJgQcWm+E1NSXnnJN84Qv9PCEAQ0FfO9Tl9wDAkNXTOfTdvt7eXhxT1xH0P/+5oAdg0Ln8HgCgtzqC/swzi6A/77zkc58reyoAhiAr9QDAkNTTKv1839fennzpS51Bf/75gh6A0lipBwCoVnt7st9+xWZ4w4YlF1yQfPazZU8FwBAm6gEAqtHeXhxT9/OfF0F/4YXJZz5T9lQADHEuvwcAhqQed7af/X3t7ck++3QG/UUXCXoAaoKVegCA7sycmey9d3Hv/LBhySWXJLvtVvZUAJBE1AMAQ8Dsm93NvkL/1LHbd39O/Q+2Sfbaq7h3fvjwIujHjRvQWQGgN5oqlUql7CFqTVtbW1paWtLa2prm5uayxwEA+qjbYJ/r8vt5wn/mzOTzny8utR8+PLnssuSTnxywWQEY2vraoaK+C6IeAOpfNUfWzfe++pkzkz33TC6+uAj6yy9Pdt21fwcEgNn0tUNdfg8AMLsZM4pz5y+9NBkxogj6XXYpeyoA6JLd7wGAhlPNKn2X75sxI9ljj86gv+IKQQ9ATbNSDwCQFEE/fnyxMj9iRHLllcnOO5c9FQB0S9QDAMyYUZw7f8UVyUILFUG/005lTwUAPXL5PQDQcOa7AV5X75s+Pdl9986gnzhR0ANQN6zUAwBDV0fQX3VVsvDCRdDvsEPZUwFA1UQ9AFDX5jlffrbvuz2n/uixyW67JVdfXQT91Vcn21e3wg8AtcI59V1wTj0A1L5ug32uy+/nCf9p04qgv+aaIuivuSbZbrsBmxUAetLXDhX1XRD1AFDbqjmybr731U+blowbl/ziF8nIkcU/t9mmX+cDgN7qa4e6/B4AGDqmTUs++cnkuuuKoL/22mTrrcueCgD6zO73AEBdqWaVvsv3TZ2a7LprEfSjRhX/FPQA1Dkr9QBA4+sI+htu6Az6rbYqeyoAWGCiHgBobG++meyyS3LTTckiiyTXX5+MGVP2VADQL1x+DwDUlflugNfV+958M/nEJzqD/oYbBD0ADcVKPQDQmN58M/n4x5NbbimC/sYbk803L3sqAOhXoh4AqFnznC8/2/fdnlN/xJhk552TW29N3vKWIug322wAJwWAcjinvgvOqQeAcnUb7HNdfj9P+L/xRhH0t91WBP1NNyUf/ehAjQoA/aKvHSrquyDqAaA81RxZN9/76l9/Pdlpp+T225NFFy2CftNN+3lCAOh/fe1Ql98DAI3h9deTj30sueOO5K1vTW6+OfnIR8qeCgAGlN3vAYCaUc0qfZfve/31ZMcdO4P+llsEPQBDgqgHAOrb//6X7LBDcuedyWKLFZvjbbxx2VMBwKBw+T0AUL86gv7uuzuDfsMNy54KAAaNlXoAoGbMdwO8rt732mvJdtsVQd/cXOx2L+gBGGKs1AMA9acj6O+9tzPo11+/7KkAYNBZqQcAakpPq/VPHb5psu22RdC3tCSTJgl6AIYsK/UAQGlm38V+9pjv+H6e1199tQj6X/2qM+jXW2/wBgaAGtNUqVQqZQ9Ra9ra2tLS0pLW1tY0NzeXPQ4ANJzujq6b70p9W1sR9Pffn7ztbUXQr7vuwAwIAIOsrx3q8nsAYFD1dBZ9l6+3tSXbbFME/dvfXpxHL+gBQNQDADWutTXZeuvkgQeKoL/99mSddcqeCgBqgqgHAAZNT6v087zvlVeSsWOTX/86ecc7ihV6QQ8As9goDwCoTR1B/7vfJe98ZxH0H/xg2VMBQE0R9QBAzWl+87Vkq62SBx8U9ADQDZffAwCDpqcz6JMi6P9w34+LoF988eTOOwU9AMyHlXoAoGa0vPFqLp7wneS5JzuD/gMfKHssAKhZoh4AGBCzb4o3+wr9U8du3+WGeS1vvJpLJnwnaz73ZPKudxVBv+aagzIrANSrpkqlUuntD7W3t+fvf/97nn/++bS3t8/x2qabbtpvw5Wlra0tLS0taW1tTXNzc9njAEBd6W6H+7kvv+9479veaMsj9/woeeSRZIkliqBfY42BHBMAakpfO7TXUf/rX/86u+++e/71r39l7h9tamrKzJkze/NxNUnUA0DfVHNk3Tz31b/wQrLllsmjjyZLLlkE/eqrD9CEAFCb+tqhvb78fr/99su6666bG2+8MUsvvXSampp6+xEAAIUXXkjGjEn+8Ici6O+6K1lttbKnAoC60euV+kUXXTSPPvpo3v/+9w/UTKWzUg8AvVfNKn2Hp47dPvnvf4ug/+Mfk6WWKoJ+1VUHcEIAqF197dBeH2m3/vrr5+9//3tvfwwAoNPzzydbbFEE/dJLJ3ffLegBoA+quvz+D3/4w6zvDzzwwBxyyCGZMmVKPvCBD2ShhRaa471rrbVW/04IADSUd/7vlSLo//zn5N3vLlboV1657LEAoC5Vdfn9sGHD0tTUNM/GeLM+5P+/ZqM8ABjaeroEf/H/vZwH7zwmeeyx5D3vKYJ+pZUGaToAqF0DulHeP//5zz4PBgCQJO967eVcevm3kxefKYL+7ruTBt6jBwAGQ1VRv/zyy8/6/p577slGG22UESPm/NEZM2bk/vvvn+O9AEDjmn1VvuOYuqeO3b7L1fp3vfZSLrvs23n/S/9OllmmWKEX9ACwwHq9+/3w4cMzefLkLLHEEnM8/+KLL2aJJZZw+T0ANLjuLrGf/Qz6jve967WX8rvbf5A8/niy7LJF0L/vfQM+JwDUk0E7p77j3vm5vfjii1l00UV7+3EAQB3p6Z75FQ69cY5V+/znP8nmmyd/+1uy3HJF0L/3vYMxKgAMCVVH/Sc+8YkkxaZ4e+65Z0aOHDnrtZkzZ+YPf/hDNtpoo/6fEACoT88+WwT9E08kyy9fBP2KK5Y9FQA0lKqjvqWlJUmxUr/YYotlkUUWmfXawgsvnA022CD77LNP/08IANSEnlbpZ3/fUwd8sAj6v/+9CPq7705WWGFA5wOAoajqqD/vvPOSJCussEK+/vWvu9QeAOjSUm0vJJttljz5ZBHyd99dhD0A0O96fU/99773vYGYAwBoAEu3/TeXXfbt5JXJxaX2d90l6AFgAPU66ldcccUuN8rr8I9//GOBBgIAatP8jqvr8O6253PZZd/O8q9MKTbDu+uuYnM8AGDA9DrqDzrooDkeT58+PQ8//HBuueWWfOMb3+ivuQCAOvKe1udz2WWHZbnW54rj6u66qzi+DgAYUL2O+q9+9atdPn/aaaflwQcfXOCBAIDaMPuq/OzH1M29Wr9M63O57LJvZ9mOoL/77mSZZQZzVAAYspoqlUqlPz7oH//4Rz70oQ+lra2tPz6uVG1tbWlpaUlra2uam5vLHgcABlV3l9h3xH3H+5ZpfS6XX3pYlml7PllppWKF/j3vGYwxAaCh9LVDh/XXAFdddVXe8Y539NfHAQAl6OnYujlW77+4eu676UhBDwAl6vXl92uvvfYcG+VVKpVMmTIl//3vf3P66af363AAQI36xz+Kc+iffjpZeeUi6N/97rKnAoAhp9dRv/POO8/xeNiwYXnXu96VzTbbLKuuump/zQUADLKeVuk7bPrFc3LPzUclzzyTrLJKEfRLLz3A0wEAXelV1M+YMSMrrLBCtt566yy11FIDNRMAUKOWf/k/xTn0r76QrLpqEfT+nQAAStOre+pHjBiRL33pS5k6depAzQMA1KjlX/5PLr/0sLz71ReS1VYT9ABQA3q9Ud7666+fhx9+eCBmAQBKNPvO9nNb4aVnM+HSQ7P0ay8mq68u6AGgRvT6nvovf/nLOeSQQ/Lvf/87o0ePzqKLLjrH62uttVa/DQcAlG/Fl57N5ZcdliVfeymPL75cVrnrrmSJJcoeCwBIL86p/8IXvpCTTjopb3vb2+b9kKamVCqVNDU1ZebMmf0946BzTj0AjW6Oo+nmWqGf/bX3vvjvXHb5t7Pkay/lr4svn1X//FtBDwADoK8dWnXUDx8+PJMnT84bb7zR7fuWX375qn95rRL1ADSq7na4nzvux+xzZi677NtZ4n8vJx/4QHLHHcm73jXQIwLAkNTXDq368vuO9m+EaAeAoainI+tWOPTGzrD/y19yx/VHJP97OVlrrSLoF1984IcEAHqlVxvlNTU1DdQcAECteOyxZPPNk+eeSz74QUEPADWsVxvlrbzyyj2G/UsvvbRAAwEA/a+nVfoOW+11eibdcGTy/PPJhz6U3H578s53DuxwAECf9SrqjzzyyLS0tAzULABAiVb+71O59PLDk9dbk7XXLoL+He8oeywAoBu9ivrddtstS9jxFgAazir/fSqXXH54Fn+9NVlnnWTSJEEPAHWg6nvq3U8PAPVr7p3tZ7fq8//MpZd9uwj60aOt0ANAHak66qs8+Q4AqCOrPf+PXHr54XnnG215dKmViqB/+9vLHgsAqFLVl9+3t7cP5BwAQD+afWO8jlX6p47dfo7nV3/uH7l4wnfyjjfa8sjSK+VDj/02edvbBntUAGABNFUswc+jra0tLS0taW1tTXNzc9njAEDVutvlfvZL8Lf//Cm5+PLv5O1vvpqsv35y662JzXABoDR97dBenVMPANSuno6tm/X673+fG689ogj6DTYQ9ABQx0Q9AAwlDz2UbLll8vLLyYYbCnoAqHOiHgAaQE+r9EnygclPpHXjjxZBv9FGyS23JG4zA4C6JuoBYAhYa/LfcsmE76Rl6v+SjTcW9ADQIEQ9ADS4D/7n8Vw84btpnvq//HaZ1ZObb04WW6zssQCAfiDqAaABzL6z/ew+9J/Hc9H/D/rfLLNGPvyX3wh6AGggoh4AGtTaz/41F034TpqnvZ7fLLtmPv/JI5K3vrXssQCAfiTqAaAOrXDojbO+Osy+Wr/Os3/JhVd8N4tNeyMPLPeB7LnrEXnsJ7uWMSoAMICaKpVKpewhak1bW1taWlrS2tqaZpsIAVBDutvlflbU/+pXeW2LrfLWaW/k/uXWykaP3Z8suuggTQgA9EVfO9RKPQDUiZ6OrVvh0BuT++5Lttkmb532RrLFFtnoLw8IegBoYKIeABrEes/8Kdlmm+S115IxY5Lrr0/e8payxwIABtCIsgcAAHrW0yr9h5/5U8678ohk+pvJVlsl116bLLLI4AwHAJTGSj0A1Ln1n/5jzr/ye1l0+pu5Z4W1BT0ADCGiHgDq2AZP/yHnXXVE3jJ9an654jrZ5xPfEfQAMISIegCoA7MfV9dhw389mvOuPDJvmT41d684Ovt+4jt5/IRPlDAdAFAW99QDQI2a3xn0SbLRU4/k5xOPziIzpubO966bL33825k6YuHBHhEAKJmoB4Aa09WmeLM/t/FTj+TnE4/KqBnTcsf71suXdv52po1YqMvVfACgsYl6AKghPe1y/5F/PpyLr/tBMmNabn/fevnyzt/O347feXCGAwBqjqgHgDqxyT9/n3MmHp3MnJ7suGO2vPLK/G3kyLLHAgBKZKM8AKgR3a3Sb/qPh3LOxKMzcub03LbSBslVVyWCHgCGPFEPADVusycfzM+u/n5GzpyeW1faIPvv9K1kYZviAQAuvweAmrbZk7/LWdf8ICNnzsgtK2+YAz72rcwY7r++AYCClXoAqBFz716/+WxBf9PKG80KervcAwAd/E/9AFCDxvz9NznjmmOycPuM3LjKxvnqjt+wQg8AzMO/HQBASWbfGK9j9f2pY7fP3rv8X07/RRH0N6zykRy049dnBb1VegBgdqIeAAZZV7vcdzz31IYzcs71xybtM3L9qpvkoB2/npnDhot5AKBLoh4ABlF3x9aN/dsDmf7jY7NQ+8xkt92y40UXZccR/qsaAJg/G+UBQA3Y+m/357Rr/3/Qf/rTyUUXJYIeAOiBqAeAQTK/VfptHv9VfnrtcVmofWauWX2zvG+Z3QQ9AFAV/8YAACXa9q/35dTrfpQRlfZcvcbm+fp2B6V92PCyxwIA6oSVegAoyXazBf1EQQ8A9IGoB4BBMvsO9jv85Z6c8v+D/qo1x+QbswW9ne4BgGq5/B4ABtmOj/0yJ91wQoZX2nPFB7bModscaIUeAOgTUQ8AA2T2jfE6Vt+fWqs1M39UBP2ED2yVQ7c9MJWmYfO8DwCgGqIeAPpZV7vcr3Dojdnpz3fl5JtOzPBKe7L33jn0HR+bFfRiHgDoC1EPAP1ofsfW7fznu3LCjScmlfZkn32SM8/MP4fZ2gYAWDD+bQIABtgn/nRHfnLDTzK80p5LP7hNcuaZiaAHAPqBf6MAgH7S1Sr9Ln+8I8ffeFKGpZKLP7RtDt/6y1nh2zeXMB0A0Ihcfg8AA+STf5iU424+JcNSyUVrb5fvbvWlpKmp7LEAgAYi6gFgAHzyD7fluJtPzbBUcsE62+d7W+4n6AGAfufyewDoJx072I979Nb8+P+v0J83esd5gt5O9wBAf7FSDwD9aLdHbsmxt/40SXLe6B1z5Jh9rdADAANG1APAAph9c7yn3vvsrKA/d/THctSYfeYJeqv0AEB/EvUA0Adz73T/mYdvSo47vXjwta/lCyeckKMOu2nW62IeABgIoh4AemnuoP/s72/M9yedkSQ5e72P54cLbZGnmpqEPAAw4EQ9ACyA8b+/IUdPOjNJctaHP5FjNvu8e+gBgEFj93sA6IXZV+k/99D1s4L+zPV3mSPo517NBwAYCFbqAaAP9nzwuhxxx9lJktM32DU/2vRzVugBgEEn6gGgl77wu2vzf3f+LEly2gafzI833UPQAwClEPUA0AtPLfm35P8H/akbjssJm3y2y6C3SR4AMBhEPQD0oOP++L1/e3W+c9e5SZKTN9otJ37kM1boAYBSiXoAmI/ZN7vb5zdX5/C7O4L+0znxI7vPN+it0gMAg0XUA0AXZg/6L/7mqhx29/lJkhM33j0nf2T3JEW8z/4+MQ8ADDZRDwDd+NKvr8y3fnlBkuQnH/lMTtn403O8LuQBgDI5px4A5tKx+v7lB66YFfQndBH0zqIHAMpmpR4AurD//RPyjXsvSpL8eJPxOW2jcSVPBAAwL1EPAHM58FeX5ZD7LkmS/GjTPXL6hp8qeSIAgK6JegCY3VFHzQr64z76uZyxwSfn+1b30wMAZRP1ANDhiCOSI49Mkhyz2Z45a/1dy50HAKAHNsoDgEolJ2+8+6ygz49/nMPuOq/bH7FKDwDUAiv1AAxpK3zrhnztvkvy1fsvT5J8f/Mv5JwXVksOvXFWuDuLHgCoVU2VSqVS9hC1pq2tLS0tLWltbU1zc3PZ4wAwQFb41g055N6Lc+ADE5IkR2+xd36+3s5zvEfEAwCDoa8daqUegKGpUsk37rkw+//6yiTJUVvsk3PX26nkoQAAesc99QAMPZVKTt/oU7OC/ogx+8436Ge/9B4AoNZYqQdgaKlUkkMPzZd/fVWS5P+2/GIuHL1jyUMBAPSNqAdg6KhUkm9+Mzn++CTJd7faLxets0PJQwEA9J2oB2BoqFSSb3wjOeGE4vFpp+Wip5fv8cdslAcA1DL31APQ+CqV5JBDZgX94WO/nHz5yyUPBQCw4EqP+tNPPz0rrrhiRo0aldGjR+fee++t6ud+9atfZcSIEfnQhz40z2sTJ07M6quvnpEjR2b11VfPNddc089TA1A3KpWcu97OyYknJkkO2/qAXLL2dlVtgGeVHgCodaVG/YQJE3LQQQfl8MMPz8MPP5xNNtkk2267bZ5++uluf661tTV77LFHxowZM89rDzzwQMaNG5fx48fn0Ucfzfjx4/OpT30qv/nNbwbqzwCgVlUqOW+9nfKFh65Lkhy69QG57EPbzPO2ueP9qWO3F/QAQF1oqlQqlbJ++frrr5911lknZ5xxxqznVltttey888455phj5vtzu+22W1ZaaaUMHz48v/jFL/LII4/Mem3cuHFpa2vLzTffPOu5bbbZJm9/+9tz2WWXVTVXW1tbWlpa0tramubm5t7/YQCUr1JJvvKV5Kc/TXuacug2B+aKD47t8q0CHgAoW187tLSV+mnTpuWhhx7K2LFz/gvW2LFjc//998/3584777w8+eST+d73vtfl6w888MA8n7n11lt3+5lTp05NW1vbHF8A1LFKJTnwwFlB/61t5x/0ibPoAYD6VVrUv/DCC5k5c2aWXHLJOZ5fcsklM2XKlC5/5oknnsihhx6aSy65JCNGdL1x/5QpU3r1mUlyzDHHpKWlZdbXsssu28u/BoCa0d6e7L9/ctpp/z/ov5Ir15p/0AMA1LPSN8pramqa43GlUpnnuSSZOXNmdt999xx55JFZeeWV++UzOxx22GFpbW2d9fXMM8/04i8AoGZ0BP0ZZyRNTfnGdgflyrW2KnsqAIABU9o59YsvvniGDx8+zwr6888/P89Ke5K8+uqrefDBB/Pwww/ngAMOSJK0t7enUqlkxIgRue2227LFFltkqaWWqvozO4wcOTIjR47sh78KgNK0tydf+lJy9tlJU1Ny/vk5YY89MtEu9wBAAyttpX7hhRfO6NGjM2nSpDmenzRpUjbaaKN53t/c3Jw//vGPeeSRR2Z97bfffllllVXyyCOPZP3110+SbLjhhvN85m233dblZwLQINrbk/32K4J+2LDkwguTPfYoeyoAgAFX2kp9khx88MEZP3581l133Wy44YY5++yz8/TTT2e//fZLUlwW/+yzz+bCCy/MsGHDsuaaa87x80sssURGjRo1x/Nf/epXs+mmm+a4447LTjvtlGuvvTa333577rvvvkH92wAYJO3tyb77Jj//eTJsWL663ddy7Z/enhx646yj6brbCM8qPQBQz0qN+nHjxuXFF1/MUUcdlcmTJ2fNNdfMTTfdlOWXXz5JMnny5B7PrJ/bRhttlMsvvzzf+c538t3vfjfve9/7MmHChFkr+QA0kPb2ZJ99knPPzcymYfna9gfnutU3m/VyR8x3hPvscS/mAYBGUOo59bXKOfUAdWDmzGTvvZPzz8/MpmE5aIdDcv3qH53v20U8AFDL+tqhpa7U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227J1VcXQX/11cn2jRG4LrsHAKBeiXqgZ9OmFUF/zTVF0F9zTbLddmVPBQAAQ56oB7o3bVoyblzyi18kI0cW/9xmm7KnAgAAIuqB7kyblnzyk8l11xVBf+21ydZblz1VadxPDwBArRH1QNemTi2C/vrrk1GjiqAfO7bsqfqVe+kBAKh3oh6Y19Spya67JjfcUAT9ddclW21V9lT9yhF2AAA0AlEPzOnNN5NddkluuqkI+uuvT7bcsuypSiPmAQCoZcPKHuD000/PiiuumFGjRmX06NG599575/ve++67LxtvvHHe+c53ZpFFFsmqq66aE088cY73nH/++Wlqaprn68033xzoPwXq35tvJp/4RBH0iyxSrNQ3YNC77B4AgEZR6kr9hAkTctBBB+X000/PxhtvnLPOOivbbrttHnvssSy33HLzvH/RRRfNAQcckLXWWiuLLrpo7rvvvnzxi1/Moosumn333XfW+5qbm/P444/P8bOjRo0a8L8H6tqbbyYf/3hyyy1F0N94Y7L55mVPBQAAdKPUqP/JT36SvfbaK3vvvXeS5KSTTsqtt96aM844I8ccc8w871977bWz9tprz3q8wgor5Oqrr8699947R9Q3NTVlqaWWGvg/ABrFG28kO++c3HZb8pa3FEG/2WZlTwUAAPSgtMvvp02bloceeihj59pNe+zYsbn//vur+oyHH344999/fz760Y/O8fxrr72W5ZdfPssss0x22GGHPPzww91+ztSpU9PW1jbHFwwZb7yR7LRTZ9DfdJOg///cTw8AQK0rLepfeOGFzJw5M0suueQczy+55JKZMmVKtz+7zDLLZOTIkVl33XWz//77z1rpT5JVV101559/fq677rpcdtllGTVqVDbeeOM88cQT8/28Y445Ji0tLbO+ll122QX746BevP568rGPJZMmJYsumtx8czLX/0jWSFY49Eb30wMA0FBK3/2+qalpjseVSmWe5+Z277335rXXXsuvf/3rHHrooXn/+9+fT3/600mSDTbYIBtssMGs92688cZZZ511cuqpp+aUU07p8vMOO+ywHHzwwbMet7W1CXsaX0fQ33FHZ9BvsknZUw0YR9gBANCISov6xRdfPMOHD59nVf7555+fZ/V+biuuuGKS5AMf+ECee+65HHHEEbOifm7Dhg3Leuut1+1K/ciRIzNy5Mhe/gVQx15/Pdlxx+TOO5O3vrUI+o98pOypaoKYBwCgnpR2+f3CCy+c0aNHZ9KkSXM8P2nSpGy00UZVf06lUsnUqVO7ff2RRx7J0ksv3edZoaH873/JDjsUQb/YYsmttzZ80LvkHgCARlXq5fcHH3xwxo8fn3XXXTcbbrhhzj777Dz99NPZb7/9khSXxT/77LO58MILkySnnXZalltuuay66qpJinPrjz/++Bx44IGzPvPII4/MBhtskJVWWiltbW055ZRT8sgjj+S0004b/D8Qas3//pdsv33yy192Bv2GG5Y9FQAA0EelRv24cePy4osv5qijjsrkyZOz5ppr5qabbsryyy+fJJk8eXKefvrpWe9vb2/PYYcdln/+858ZMWJE3ve+9+XYY4/NF7/4xVnveeWVV7LvvvtmypQpaWlpydprr5177rknH/7whwf974Oa8tprRdDfc0/S3FwE/Wz7TwAAAPWnqVKpVMoeota0tbWlpaUlra2taW5uLnscWHCvvZZst11y771F0N92W7L++mVPNWiqvfze/fQAAJSlrx1a+u73wAB79dUi6O+7L2lpKYJ+iFy54l56AAAanaiHRvbqq8m22ya/+lUR9JMmJeutV/ZUg8IRdgAADAWiHhpVW1sR9Pffn7ztbUXQr7tu2VPVHDEPAEA9E/XQiFpbk222SX796+Ttby+CfvTosqcaNC67BwBgqBD10GhaW5Ott05+85si6G+/PVlnnbKnAgAABsCwsgcA+tErryRjxxZB/453JHfcIegBAKCBWamHRtER9L/7XWfQf+hDZU9V09xPDwBAvRP10AhefrkI+gcfTN75ziLoP/jBsqcadO6lBwBgqBH1UO9efjnZaqvkoYeSxRcvgn6ttcqeatA5wg4AgKFI1EM9e+mlIuh///si6O+8M/nAB8qeqqaJeQAAGomoh3r10kvJllsmDz+cvOtdRdCvuWbZU5XCZfcAAAxVdr+HevTii8mYMUXQL7FEctddQzboAQBgKBP1UG9eeKEI+kceSZZcsgj6NdYoeyoAAKAELr+HetIR9H/4Q2fQr7Za2VPVDffTAwDQaEQ91Iv//rcI+j/+MVlqqSLoV1217KlK5V56AACGOpffQz14/vlkiy2KoF966eTuuwW9I+wAAMBKPdS8jqD/85+Td7+7WKFfeeWyp6obYh4AgEYm6qGWPfdcEfSP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qVE/YcKEHHTQQTn88MPz8MMPZ5NNNsm2226bp59+usv3L7roojnggANyzz335C9/+Uu+853v5Dvf+U7OPvvsWe954IEHMm7cuIwfPz6PPvpoxo8fn0996lP5zW9+M1h/FgAAAAyKpkqlUinrl6+//vpZZ511csYZZ8x6brXVVsvOO++cY445pqrP+MQnPpFFF100F110UZJk3LhxaWtry8033zzrPdtss03e/va357LLLqvqM9va2tLS0pLW1tY0Nzf34i8CAACA3utrh5a2Uj9t2rQ89NBDGTt27BzPjx07Nvfff39Vn/Hwww/n/vvvz0c/+tFZzz3wwAPzfObWW2/d7WdOnTo1bW1tc3wBAABArSst6l944YXMnDkzSy655BzPL7nkkpkyZUq3P7vMMstk5MiRWXfddbP//vtn7733nvXalClTev2ZxxxzTFpaWmZ9Lbvssn34iwAAAGBwlb5RXlNT0xyPK5XKPM/N7d57782DDz6YM888MyeddNI8l9X39jMPO+ywtLa2zvp65plnevlXAAAAwOAbUdYvXnzxxTN8+PB5VtCff/75eVba57biiismST7wgQ/kueeeyxFHHJFPf/rTSZKlllqq1585cuTIjBw5si9/BgAAAJSmtJX6hRdeOKNHj86kSZPmeH7SpEnZaKONqv6cSqWSqVOnznq84YYbzvOZt912W68+EwAAAOpBaSv1SXLwwQdn/PjxWXfddbPhhhvm7LPPztNPP5399tsvSXFZ/LPPPpsLL7wwSXLaaadlueWWy6qrrpqkOLf++OOPz4EHHjjrM7/61a9m0003zXHHHZeddtop1157bW6//fbcd999g/8HAgAAwAAqNerHjRuXF/9fe/cfU1X9x3H8dfXCvYSipgHXoUzBULMSJBXwR0rzR/YHqy3LRpo6fyxX/uEPHCDmcmGzmT8mm02lmT8X6lyRP4v5W8PAmZIYw+USwiwQbCjznu8fzTsvYF/Ae7nee5+P7W6ecz73c+65e+0t73vOvefWLS1fvlwVFRUaNGiQ8vPzFRkZKUmqqKhwume93W7XkiVLVF5eLrPZrKioKGVnZ2v27NmOMYmJidq5c6cyMjKUmZmpqKgo7dq1S8OGDWv34wMAAAAAwJ08ep/6J1VNTY26du2q69evc596AAAAAIDb3b59W7169VJ1dbW6dOnS4ud59Ez9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\n",
      "text/plain": [
       "<Figure size 1200x1200 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12, 12))\n",
    "plt.scatter(hf_mean_lin_mf_model, hf_mean_high_gp_model, alpha=0.1)\n",
    "min_max = [hf_mean_high_gp_model.min(),hf_mean_high_gp_model.max()]\n",
    "plt.plot(min_max, min_max, color='r')\n",
    "plt.ylabel('Truth')\n",
    "plt.xlabel('Prediction');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 263,
   "id": "a8e61a10",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " /home/mech/smdsouza/.local/lib/python3.9/site-packages/seaborn/axisgrid.py:1826: UserWarning:\n",
      "\n",
      "`shade_lowest` has been replaced by `thresh`; setting `thresh=0.05.\n",
      "This will become an error in seaborn v0.13.0; please update your code.\n",
      "\n",
      " /home/mech/smdsouza/.local/lib/python3.9/site-packages/seaborn/axisgrid.py:1826: FutureWarning:\n",
      "\n",
      "`shade` is now deprecated in favor of `fill`; setting `fill=True`.\n",
      "This will become an error in seaborn v0.14.0; please update your code.\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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zEVISUkhODiYpKQkLcYpIuIEMjMzCQqLJCc92a7XDQgIIDU11Wbb+PHjmTBhQrmuZ7FYGD58OE899RRt2rS56PFpaWnMmzePRo0aUb9+/WKPS0pKIjQ0tFw16T3PvhSsRETE6WVnZ5OTnkznuyfh7uVjl2uaszPZtOA5jh49ahM4KrLszZQpU/Dw8OCxxx4r8bhZs2bx9NNPk5aWRsuWLVmxYgVeXl5FHrt//35mzpzJm2++We66xH4UrERExGW4e/ng4eVr12sGBQXZpSVn06ZNzJgxg82bN1+0y23YsGH079+fuLg4pk6dym233ca6devw8bENjcePH+eaa67h1ltv5f77769wjVJxGrwuIiJSBdasWUNCQgINGjTAw8MDDw8PDh8+zBNPPEHDhg1tjg0ODqZZs2b06tWLL774gl27drF06VKbY44fP07fvn3p1q0bc+fOrcI7kZKoxUpERKQKDB8+nKuuuspm29VXX83w4cO55557SjzXMAyysrLyv4+NjaVv3775nyB0c1M7SXWhYCUiImInqamp7Nu3L//7gwcPsnXrVkJDQ2nQoAFhYWE2x3t6ehIVFUWLFi0AOHDgAIsXL2bAgAGEh4cTGxvLlClT8PX1ZeDAgYC1papPnz40aNCAqVOncvLkyfzrRUVFVcFdSkkUrEREROxk48aN9O3bN//7MWPGAHD33Xczf/78i57v4+PDmjVrmD59OmfPniUyMpJevXqxfv16IiIiAPjxxx/Zt28f+/bto169ejbnawYlx9M8VkVITk7WR09FRJxI3t/tyx6YZrfB67nZGfzx/hiXfy/Qe559qVNWRERExE4UrERERETsRMFKRERExE4cHqxmzZpFo0aN8PHxoXPnzqxZs6bYY3/55RdMJlOhR8FVv+fPn1/kMZmZmVVxOyIiIlKDOfRTgYsXL2b06NHMmjWLHj16MGfOHK699lr++ecfGjRoUOx5u3fvthlgFx4ebrM/KCiI3bt322y7cLZaEREREXtzaLCaNm0a9913X/40/NOnT2f58uXMnj2byZMnF3teREQEISEhxe43mUxlmssjKyvLZuK15GT7LuIpIiJSXeg9r3I5rCswOzubTZs2MWDAAJvtAwYMYP369SWe27FjR+rUqUO/fv1YtWpVof2pqanExMRQr149rr/+erZs2VLi9SZPnkxwcHD+o6QV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\n",
      "text/plain": [
       "<Figure size 600x600 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "df=pd.DataFrame({'25RANS5DNS': hf_mean_high_gp_model.flatten(),'Reference':hf_mean_lin_mf_model.flatten()})\n",
    "rel=sns.jointplot(x=\"25RANS5DNS\", y=\"Reference\", kind = \"kde\",data = df, fill=True, n_levels = 6, shade = True, cbar = True, shade_lowest = False)\n",
    "rel.fig.suptitle(\"25RANS + 5 DNS\")\n",
    "plt.plot(min_max, min_max, color='r')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa45a305",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3be2dc52",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
