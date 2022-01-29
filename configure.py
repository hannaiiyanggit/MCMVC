def get_default_config(data_name, mark):
    if data_name in ['Caltech101-20']:
        """The default configs."""
        config = dict(Prediction=dict(
                    arch1=[128, 256, 128],
                    arch2=[128, 256, 128],
                ),
                Autoencoder=dict(
                    arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                    arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                    activations1='relu',
                    activations2='relu',
                    batchnorm=True,
                ),
                Classifier=dict(
                    arch=[128, 256, 128, 64, 1],
                    activations='leakyrelu',
                ))
        if mark==0:
            config['training']=dict(
                    seed=4,
                    batch_size=256,
                    epoch=500,
                    lr=1.0e-4,
                    # Balanced factors for L_cd, L_pre, and L_rec
                    alpha=9,
                    lamda1=0.2,
                    lamda2=0.1,
                    lamda3=0.2,
                    lamda4=0.2
                )
        elif mark==1:
           config['training']=dict(
                    seed=4,
                    batch_size=256,
                    epoch=1000,
                    lr=1.0e-4,
                    # Balanced factors for L_cd, L_pre, and L_rec
                    alpha=9,
                    lamda1=0.2,
                    lamda2=0.1,
                    lamda3=0.2,
                    lamda4=0.2,
                    mu = 0.2
                )
        elif mark==2:
            config['training']=dict(
                    seed=4,
                    batch_size=256,
                    epoch=500,
                    lr=1.0e-4,
                    # Balanced factors for L_cd, L_pre, and L_rec
                    alpha=9,
                    lamda1=0.1,
                    lamda2=0.1,
                    lamda3=0.2,
                    lamda4=0.2
                )
        elif mark==3:
            config['training']=dict(
                    seed=4,
                    batch_size=256,
                    epoch=1000,
                    lr=1.0e-4,
                    # Balanced factors for L_cd, L_pre, and L_rec
                    alpha=9,
                    lamda1=0.2,
                    lamda2=0.1,
                    lamda3=0.3,
                    lamda4=0.2,
                )
        else:
            raise Exception('Undefined loss function or missing rate out of range')
        return config
    elif data_name in ['Scene_15']:
        """The default configs."""
        config=dict(
                Prediction=dict(
                    arch1=[128, 256, 128],
                    arch2=[128, 256, 128],
                ),
                Autoencoder=dict(
                    arch1=[20, 1024, 1024, 1024, 128],
                    arch2=[59, 1024, 1024, 1024, 128],
                    activations1='relu',
                    activations2='relu',
                    batchnorm=True,
                ),
                Classifier=dict(
                    arch=[128, 256, 128, 64, 1],
                    activations='leakyrelu',
                ))
        if mark==0:
            config['training']=dict(
                    seed=8,
                    batch_size=256,
                    epoch=400,
                    lr=1.0e-3,
                    alpha=9,
                    lamda1=0.1,
                    lamda2=0.1,
                    lamda3=0.3,
                    lamda4=0.2
                )
        elif mark==1:
            config['training']=dict(
                    seed=8,
                    batch_size=256,
                    epoch=500,
                    lr=1.0e-3,
                    alpha=9,
                    lamda1=0.2,
                    lamda2=0.1,
                    lamda3=0.1,
                    lamda4=0.2,
                )
        elif mark==2:
            config['training']=dict(
                    seed=8,
                    batch_size=256,
                    epoch=300,
                    lr=1.0e-3,
                    alpha=9,
                    lamda1=0.1,
                    lamda2=0.1,
                    lamda3=0.7,
                    lamda4=0.2
                )
        elif mark==3:
            config['training']=dict(
                    seed=8,
                    batch_size=256,
                    epoch=500,
                    lr=1.0e-3,
                    alpha=9,
                    lamda1=0.1,
                    lamda2=0.1,
                    lamda3=0.5,
                    lamda4=0.2,
                )
        else:
            raise Exception('Undefined loss function or missing rate out of range')
        return config
    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        config=dict(
                Prediction=dict(
                    arch1=[128, 256, 128],
                    arch2=[128, 256, 128],
                ),
                Autoencoder=dict(
                    arch1=[784, 1024, 1024, 1024, 64],
                    arch2=[784, 1024, 1024, 1024, 64],
                    activations1='relu',
                    activations2='relu',
                    batchnorm=True,
                ),
                Classifier=dict(
                    arch=[64, 128, 64, 32, 1],
                    activations='leakyrelu',
                ))
        if mark==0:
            config['training']=dict(
                    seed=0,
                    epoch=650,
                    batch_size=256,
                    lr=1.0e-3,
                    alpha=9,
                    lamda1=0.1,
                    lamda2=0.1,
                    lamda3=0.3,
                    lamda4=0.2
                )
        elif mark==1:
            config['training']=dict(
                    seed=0,
                    epoch=300,
                    batch_size=256,
                    lr=1.0e-3,
                    alpha=9,
                    lamda1=0.3,
                    lamda2=0.1,
                    lamda3=0.4,
                    lamda4=0.2
                )
        elif mark==2:
            config['training'] = dict(
                seed=0,
                epoch=500,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lamda1=0.1,
                lamda2=0.1,
                lamda3=1.0,
                lamda4=0.2
            )
        elif mark==3:
            config['training'] = dict(
                seed=0,
                epoch=200,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lamda1=0.1,
                lamda2=0.1,
                lamda3=1.0,
                lamda4=0.2
            )
        else:
            raise Exception('Undefined loss function or missing rate out of range')
        return config
    elif data_name in ['LandUse_21']:
        """The default configs."""
        config = dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 64],
                arch2=[40, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            Classifier=dict(
                arch=[64, 128, 64, 32, 1],
                activations='leakyrelu',
            ))
        if mark==0:
            config['training']=dict(
                seed=3,
                epoch=500,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lamda1=0.5,
                lamda2=0.1,
                lamda3=0.2,
                lamda4=0.2
            )
        elif mark==1:
            config['training'] = dict(
                seed=3,
                epoch=400,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lamda1=1.1,
                lamda2=0.1,
                lamda3=1.1,
                lamda4=0.2
            )
        elif mark==2:
            config['training'] = dict(
                seed=3,
                epoch=700,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lamda1=0.1,
                lamda2=0.1,
                lamda3=1.0,
                lamda4=0.2
            )
        elif mark==3:
            config['training'] = dict(
                seed=3,
                epoch=700,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lamda1=0.1,
                lamda2=0.1,
                lamda3=0.7,
                lamda4=0.2
            )
        else:
            raise Exception('Undefined loss function or missing rate out of range')
        return config
    else:
        raise Exception('Undefined data_name')