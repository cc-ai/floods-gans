from comet_ml import Experiment
import time
import sys
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_val_set
from util.visualizer import overlay_flood_mask
import os

comet_exp = None
comet_exp = Experiment(api_key="<api_key>",
                        project_name="<project_name>", workspace="<username>")

if __name__ == "__main__":
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print("#training images = %d" % dataset_size)
    # INFERENCE CODE 
    opt_test = opt
    opt_test.num_threads = 1        # test code only supports num_threads = 1
    opt_test.batch_size = 1         # test code only supports batch_size = 1
    opt_test.serial_batches = True  # no shuffle
    opt_test.no_flip = True         # no flip
    opt_test.display_id = -1        # no visdom display

    opt_test.ntest = float("inf")
    opt_test.aspect_ratio = 1
    opt_test.phase = 'sample'
    opt_test.num_test = 50 

    data_loader_test = CreateDataLoader(opt_test)
    dataset_test = data_loader_test.load_data()
    # INFERENCE CODE END

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result, experiment
                )
                
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, opt, losses
                    )

            if total_steps % opt.save_latest_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_steps %d)"
                    % (epoch, total_steps)
                )
                save_suffix = "iter_%d" % total_steps if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()      
            
        if epoch % opt.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_steps)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        )
        # INFERENCE CODE 
        try:os.mkdir(opt_test.results_dir+'epoch'+str(epoch)+'/')
        except:pass
        try:os.mkdir(opt_test.results_dir+'epoch'+str(epoch)+'/val_set/')
        except:pass
        try:os.mkdir(opt_test.results_dir+'epoch'+str(epoch)+'/overlay/')
        except:pass           

        for i, data in enumerate(dataset_test):
            if i >= opt_test.num_test:
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))

            save_val_set(opt_test.results_dir+'epoch'+str(epoch)+'/val_set/',img_path, visuals,aspect_ratio=opt_test.aspect_ratio, width=opt_test.display_winsize)
        # Add the transformation in blue overlay
        overlay_flood_mask(opt_test.results_dir+'epoch'+str(epoch)+'/val_set/',opt_test.results_dir+'epoch'+str(epoch)+'/overlay/')
        print('overlay is saved')
        if epoch % 10 == 0:         
            # add comet ML part where we take the img_paths, overlay and save
            if comet_exp is not None:
                fake_im_list = fake_img(opt_test.results_dir+'epoch'+str(epoch)+'/val_set/')
                for img_path in fake_im_list:
                    comet_exp.log_image(opt_test.results_dir+'epoch'+str(epoch)+'/val_set/'+img_path)
                list_img=os.listdir(opt_test.results_dir+'epoch'+str(epoch)+'/overlay/')
                for img_path in list_img:
                    comet_exp.log_image(opt_test.results_dir+'epoch'+str(epoch)+'/overlay/'+img_path)
            print('Inference is done, on validation set')

        # INFERENCE CODE END      
        model.update_learning_rate()