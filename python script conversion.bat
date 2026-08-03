setlocal
set PATH=D:\Image_Dataset\multi_view_training_dslr_undistorted;
:: Runs your command
python colmap2mvsnet_acm.py --dense_folder \courtyard --save_folder \courtyard
endlocal
pause