import torch
import torch.nn as nn

#from utils.parser import args
import utils.logger as logger
from utils.init import init_device,init_model 
from utils.parser import args
from dataset.dataset import datasetLoader1,datasetLoader2 
from utils.solver import Tester,Trainer
from utils.F_loss import Balanceloss,Diagloss
from torch.nn import MSELoss

def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)
    if args.phase==1:
        # Create the data loader
        train_loader, test_loader = datasetLoader1(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=pin_memory)()#两个括号是自调用，__call__函数  
        criterion = Diagloss().to(device)  
    else:
        train_loader, test_loader = datasetLoader2(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=pin_memory)()#两个括号是自调用，__call__函数   
        criterion = Balanceloss(rou=args.rou).to(device)    
    # Define model

    model = init_model(args)
    model.to(device)#加载模型到相应设备cpu或gpu

    #Define loss function

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion,phase=args.phase)(test_loader)
        return

    # Define optimizer and scheduler
    lr_init = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))

    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      phase = args.phase,
                      resume=args.resume)

    # Start training
    trainer.loop(args.epochs, train_loader, test_loader)

    # Final testing
    loss = Tester(model, device, criterion)(test_loader)
    logger.info(f"\n=! Final test loss: {loss:.3e}")
        
if __name__ == "__main__":
    main()