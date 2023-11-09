import qrcode
from PIL import Image

Logo_link = '/home/cwp29/Documents/phd_misc/1de7ee70-4918-4103-98b3-5b8db6572afb.jpeg'

logo = Image.open(Logo_link)

basewidth = 110

# adjust image size
wpercent = (basewidth/float(logo.size[0]))
hsize = int((float(logo.size[1])*float(wpercent)))
logo = logo.resize((basewidth, hsize), Image.ANTIALIAS)
QRcode = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_H
)

# taking url or text
url = 'https://arxiv.org/abs/2310.06096'

# adding URL or text to QRcode
QRcode.add_data(url)

# generating QR code
QRcode.make()

# taking color name from user
QRcolor = 'Grey'

# adding color to QR code
QRimg = QRcode.make_image(
    fill_color=QRcolor, back_color='white').convert('RGB')

# set size of QR code
pos = ((QRimg.size[0] - logo.size[0]) // 2,
       (QRimg.size[1] - logo.size[1]) // 2)
QRimg.paste(logo, pos)

# save the QR code generated
QRimg.save('arxiv_qr.png')
QRimg.save('/home/cwp29/Documents/posters/dfd/figures/arxiv_qr.png')

print('QR code generated!')
