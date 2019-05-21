import base64
import zlib
import numpy as np
import struct
import matplotlib.pyplot as plt


def parse(b64_string):
    # fix the 'inccorrect padding' error. The length of the string needs to be divisible by 4.
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    # convert the URL safe format to regular format.
    data = b64_string.replace("-", "+").replace("_", "/")

    data = base64.b64decode(data)  # decode the string
    data = zlib.decompress(data)  # decompress the data
    return np.array([d for d in data])


def parseHeader(depthMap):
    return {
        "headerSize": depthMap[0],
        "numberOfPlanes": getUInt16(depthMap, 1),
        "width": getUInt16(depthMap, 3),
        "height": getUInt16(depthMap, 5),
        "offset": getUInt16(depthMap, 7),
    }


def get_bin(a):
    ba = bin(a)[2:]
    return "0" * (8 - len(ba)) + ba


def getUInt16(arr, ind):
    a = arr[ind]
    b = arr[ind + 1]
    return int(get_bin(b) + get_bin(a), 2)


def getFloat32(arr, ind):
    return bin_to_float("".join(get_bin(i) for i in arr[ind : ind + 4][::-1]))


def bin_to_float(binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]


def parsePlanes(header, depthMap):
    indices = []
    planes = []
    n = [0, 0, 0]

    for i in range(header["width"] * header["height"]):
        indices.append(depthMap[header["offset"] + i])

    for i in range(header["numberOfPlanes"]):
        byteOffset = header["offset"] + header["width"] * header["height"] + i * 4 * 4
        n = [0, 0, 0]
        n[0] = getFloat32(depthMap, byteOffset)
        n[1] = getFloat32(depthMap, byteOffset + 4)
        n[2] = getFloat32(depthMap, byteOffset + 8)
        d = getFloat32(depthMap, byteOffset + 12)
        planes.append({"n": n, "d": d})

    return {"planes": planes, "indices": indices}


def computeDepthMap(header, indices, planes):

    v = [0, 0, 0]
    w = header["width"]
    h = header["height"]

    depthMap = np.empty(w * h)

    sin_theta = np.empty(h)
    cos_theta = np.empty(h)
    sin_phi = np.empty(w)
    cos_phi = np.empty(w)

    for y in range(h):
        theta = (h - y - 0.5) / h * np.pi
        sin_theta[y] = np.sin(theta)
        cos_theta[y] = np.cos(theta)

    for x in range(w):
        phi = (w - x - 0.5) / w * 2 * np.pi + np.pi / 2
        sin_phi[x] = np.sin(phi)
        cos_phi[x] = np.cos(phi)

    for y in range(h):
        for x in range(w):
            planeIdx = indices[y * w + x]

            v[0] = sin_theta[y] * cos_phi[x]
            v[1] = sin_theta[y] * sin_phi[x]
            v[2] = cos_theta[y]

            if planeIdx > 0:
                plane = planes[planeIdx]
                t = np.abs(
                    plane["d"]
                    / (
                        v[0] * plane["n"][0]
                        + v[1] * plane["n"][1]
                        + v[2] * plane["n"][2]
                    )
                )
                depthMap[y * w + (w - x - 1)] = t
            else:
                depthMap[y * w + (w - x - 1)] = 9999999999999999999.0
    return {"width": w, "height": h, "depthMap": depthMap}


# see  https://stackoverflow.com/questions/56242758/python-equivalent-for-javascripts-dataview
# for bytes-parsing reference

# see https://github.com/proog128/GSVPanoDepth.js/blob/master/src/GSVPanoDepth.js
# for overall processing reference

# Base64 string from request to:
# https://maps.google.com/cbk?output=xml&ll=45.508457,-73.532738&dm=1
# (search https://www.google.fr/maps/place/675+Rue+Green,+Saint-Lambert,+QC+J4P+1V9/@45.5097776,-73.5063039,1093m/data=!3m2!1e3!4b1!4m5!3m4!1s0x4cc91b29203b584b:0x2f4301d5e6b04c9f!8m2!3d45.5097739!4d-73.5041152)
# -> edit long and lat for future requests
s = "eJzt13lgTAcCx_EQJJHEFZEIkqWIIwniTObNvCOoK4ljHVlXCKqoVigRyTzivou2tihVq6iuqtY5b-ZRR7eO0lJlHV1FUYu41SL75kgyM3kzmUnmSPJ-n38Skcy8N9_fm8N7hodHeY9y3h4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATlfJ3QcAblUpl7sPBNyikhl3Hw-4lnl_jEBaRPtjApKB_tKG_tKG_tKG_tKG_tKG_tJmqX_Nmu4-MnAFy_1rYgISgP7Shv7Shv7SVtL7e2lVsMbdR1i6ldD-Xias9scciqOE9fcSZU9_DMAuJaG_eHT0dwWr_X19fZ1774WWR38nK6y_cxeA_u5WeH9nDsAZ_UmBEw85V2RkpAvuxenKZv8mHTp00N5-E-cdOvo7gDP7N8nVUTDc4YeO_g6A_rbwrWg_W2_bhv6NBE46Mzf3b9NG_7V-nTp17D10oX_DYp27KF9R7u_vpAG4vX-b-jp1dBpr2Xro2v4Ni70APyPi7dG_JPe3YwB-hSnj_VuL_MwZ_QPyOK9_XFxuf1sHUGj9Mt8_Orp1a_MNOLd_QEAzA_2_ChySlf4pKdbOJS7undz-Ng7Ajf379-_WrVu8gBY5Lgf2DwwMtPbf0SIDcG1_waDOnTvn330h_YXf7Txd9FyM-zcMz2f57Et5_16iJ2XyrjnQICw2Nlb_nfGvttCKLhH9dd8IbxACAsT6KwRdunRJaaXXNJdZ_7gS3n_oUJpOTKS1jPv_NY9SOCh6yOtF7697yPL_GZbHuL9hE_r8LQZr-wvPA6Wgf-NhZv2NB9DVUv-uWmL5-7unf2iunj379qW0hO-7dQulhS-vv_568fsH5rHav0Ve_9ba_tG5GyhN_ce-VcT-iYnu6C_UdWT_1gVeu4vZX_dzt_bvYEv_Vkb9JxjO_G3J939DUMT-rZ3dP9pt_d8doSX6AuDy_m8K4uOd0X9A8a5_p_cXefTzpKZOnJg26W8GGXb176c1erTui0n_yaNHdxLn4TE1955d3X_KSC0r_fv06TPOzv7smAECH2-9ktq_gV7VqlXzu5fTCTGWmZCQUHj_hISmTRMMBup56uX39zQRUkB6vXo9eri6v3DSSUlJ-f2TtLpPE6KP66M1fvz4UVrW-vsa9R8zRh_fx4cx0D1kTB6T_vofGR58n7Z6Mi3T69EZ_f39_RuYqKrT3pRuEKL9CwY0VU-nh4nC_iYkxNX9q1cP1oky4e0d3N2Urf19cjFtxZn0l1lifEBO6v-anr-O2BLy12DWv73YRgoovLWgvJEYLdf3NxJsxGwPtvbPD01YltdfdHpaycnJ2v_KPTBn9Beeg1_L55_HwhIMtNd0O5uINi5fILgR7W27uv9fclWvbnEKWrb2z_19gvAujNDfuLVedXNyLWf0D8p_LRbdQYExFNiC2SzMWE4t8st6ERERbutvaQiGNdjQ33Qw3uYLEid2b4bo8toGnp5O6a9l-p7M8g7sW4OlSYgFN-Pq_lWqVNGfe8EhmCyhkP52Fy6suSG8gdP6i4zAdAiiO7AwikL6F4xtQv9XbuifK__0RbZgrb_4S0dRgpt3N6jrjP7lgkwUHIHdSzBV6BOEKFf3bymoYsr8IbC3v22NbciuS6_jlP46No3AfAo2bcHe8r21DH_ruv66d2AiIzAfgrX-4q8bNkYXrW7c3kD4SOmQ_qKf1WwfgegeitS_t3Fwy5q7oL-1ERiWYK2_-MNTpORi5ev6FCSaX_dOu7KB-Idya-x4KrCFzS8UltPrWbj9WloO7G99BJb6Wz5_GzKLKay8Vo08BT9UF72_g1fgvPKehvi1apkeum39q-Wy_MlcZAWW-lt532AvW8qbxBepX-z8jptBccJbKa8lWt9m1arZMAH9CFra07-Qt5HFL1_Ype_I_sWfgYOveEfVt2MBxs8Ftla3r7_t3W249J3RP1cRduDA671g_WKfkB0LsP7GoEj97exu26XvzP5GbJ2CYy52sfjFr69lz5NAUYZgtbrN0e249F3U31yQhUEU4xp3QX2dIk7Atink9ba7dREvfTf1t6TYuUXrO_ggi7kA0UkYNuGA7PbGL7v9HX7p53P0BHK5IX5Jyu_I_s6Lr-eUCTg0vm31y2R_J176Rhw_AdfHL4P9XRNfz8ETcFB829uXvf4ujK_nyAm4Pn7Z6u_y-HoOm4Dr45eh_m6Kr-eYCbi8vVH-0t3frfENir8BV7cvYZd_Efu78g1fYYq3AVe3LwP9S1D7XEXfgKvbl_L-JenCN1O0Dbg4fWnuX4Lb57F7BK4tX2r7l4b2-exYgWvLl8r-pSq9kWq2zMCF2Utifuv9S2t5M9aG4MzQpbh_GSlvgX4Kum_RX0LdRaC_tHqbk3p_qUN_aUN-aUN_aUN_aUN_aUN_aXNbfvQvEdzW390nDjroL23oL23oL23uyo_-JYO7-rv7vEEP_aUN_aUN_aUN-aUN_aUN_aUN_aXNPfnRv6RwT393nzXkQn9pQ39pQ35pQ39pQ39pQ39pQ35pQ39pQ39pQ35pQ39pQ39pQ35pQ39pQ39pQ35pQ39pQ35pQ39pQ35pQ39pc21-9C9pXNvf3WcL5pBf2tBf2pBf2tBf2pBf2pBf2tBf2pBf2pBf2pBf2tBf2pBf2pBf2pBf2lBf2pBf2pBf2pyb391nB4VxZn3kL_lQX9qclt_dJwY2QX1pc0Z81C89UF_aEF_aEF_aHBbf3ScCRYLrXtqQXtpQXtrsD-_uIwZHQnVpQ24AAAAAAAAAAAAAgLLHw0yLNf6f-HQcTFR6zvJLF4dSL9PHEntXDiBmPGT52gtDqeAFFPfB1vpE12cs_-R-GNUgleU_7j5bvbdTCMemTKe9erFk5eXvq1-99CZyRg2k3z7WhAxJo3h2WcOYsNuJtOIb4XY5Wr57iB8X4pNKV_p-NDHj022yScLt7X4UTr07tgIR-sMfssgXLO_1qg41I0T56dqKdRU_lc_kO-2oTY1fMZW8fi5FM_P2dVnrQd3oxAtTyIBXrCLcL5K4-yqDHtk3i7u67qHs-mOWb74jmPrx-mzZsSENiSbC7aVeCaOu_e0wd6xHLKG5yfJtLtWjjp4eRTxc-UJdo6KSzxkRTzXqRRP7U7pzKU9YPnthJ6rG56OJ5PNb1B8FsXyzf5NU3RNK_sTzQ-q5R-txVY-wTLOjY0nl0O80_JgvZIdVMrrXtE6yky9ZPqrRn6orTdczx-55yg6k_0edM0nJLz5wlzzFd-U-uxyiCPXJ5C9nTqR2NfUi3jvcj2gonP-3VQOpP86ckKUemkFkCPcf-bI2tek6per-7XrZ7VcsH0FXpd7PUvLXuAvqBZ5h3JOsJPpuzFrZcrqcxudhJv-UnkZRUUNksd_tl2UKv39kfk1qNcuSAwZnymv9XpuYUnkWM2u1kr8x8ow6ZF0oF3N4OT20YheizrJYTf0d0_i3Lp6ilmZ20Iz9YRrZpPkBWeyg5_QvI49xNZK_kfveUfKKPtMp9vks7gfVAPnC1Sz_Z0JH6lr3buq-21gynLkt-3f7d5jErkPI4Y9fapITs2TlLmroSZFK_mnKQ_mbi8tzYZ_9g_ZoVl1d82eWvJ55T_b9jOtM2hiWj241Xz5I48Pt7BoS17d-L_7hxgBy_2sK1aIzzZg3trXhEwa2Jb86PG7f3mHrmaOPYrlTv7_kyIssH1-pHLXj3hnurffnK7YoJ_Nf3XtAtUpLID_sFUQu9qQI1VdeTNDwOcQbl9qpz-xk-flrFlNXU09wQ26x_PmNHtyC27uYar8Ok12Zmy33i1Xyyp5byQaqGH5iVEuyxe71-zym92DGbWP5f-zton5vZDC3Zvq3TPyOF4Tvrywf-J9s1fg-15i65zP5K3UqaLYn1edGrJhO_xJVQ_ZFp057zuaw_MHOBNXadxaRNvEN4kE2y18a1ZDyXXxEdXFCgqK2Op3fcTKAjpq1Q-1dkSXrLfpV5nn_ayboSDXZuGc58vsvM_kDKz-nTjXP4Bt3bqlIm_ZYtfFORbpd5k4i7kyUZmJkBh-TFEn_9NsCom2UB7HiEcv_65PXqPf6bJc9Lxcjb8ax_MMvr5F7FrL8rdET5N6nK3OffbuOTs--o_nvF8PJeYuayR7VzGEa_RhI9v4gkf9wS_q-igTPeG0JVauPs_zlleW5Ud99yKwhrxNND8zhJl8S9iifR0VQncnf79Unh7EUcfrgIGbPrg_Vu3uwZLXjV2Ut999i5jadrsru_0ie0lzJH2sTQy9YEqO5emAa-fOp_bJzNz9mlgVflHvMVPJ8yHnVjcV7mV_K-fEpvfuRWxYfim0wQsFkbQ7h2v-PJb-nnspmBQTGjT-bKg8QzmFRzE0VsfEj5kjix0Td4fPVHZNZvn38Acr_8geq7T26yft-yfI_hs2nzrzLkg2OzVTH1K5MnHi8n_ny8CrVaG6g_PonLP_m0bVU1i4l-XPj4_J_RoYSm8Y8YjyCepLhgYFk4nmSOPfbSfr0EJbsQCyRDz8SQsQvb0RFfZLOvf3OIrXfEJYf2nEt-XDAJiLiYhX1hKMsf3zsccr_bG_5hE0sH7TuD1Vy4p_M08UZZDYdrhkW9kyWNOdzZl_LjapnC30U5D8z-XZ1J9Ebx7Lk0sHz5C8nhBDTVqqYU5vWyYff2M29Cmf5mK57qGeNb8kvpWdxS-OVPENS9PK5gZo1TWjiFpFJ3rjfhD6Rtln1_PfynK_wHHOt_C7yrcx9ii6jxpMdDjUlVo30Zupvn6FZv2EK_9ArlNuz6jHTrnpfstNNf_7FoN6y5CkHmaBtmeTkCB9F8q_1iez-1eMajLsoi6szRB68VmjusYS8V3UqHzN_nGbNoObc30_PpX7444J8c5aS7FfBj0h9rmEWCdeRfzlKnjPOj-t1qxHTJq3XPv_NgxQDX0zllwV9T7f8JlUxc28EcXF7Gv9g8gY6Pe1N4l_Vx6sXLhD-7tZlqtmhe0Sj31h-StJ91Z53g-M-kvUiercJJxRPWf7Z1ZbUweZ3iVXnd3Bbz7D8_pwDVMCTr4lWN09yNS6w_IBh31Any8eSnS-2IGd2pwhPLw1zIX4ncXrTEy72J2EjFXlq1AmW3zy-rjq6XhDnoXxKXQtvRe4JDZMfndeBzAx_RasXpZJR2zYo5uyNJrZ6JTHJY29z4VdZUp51X-bXdyvTIL0D6ZXWinzcmiJGNr3GnH3aW72aXkjMW8eSK77YTy0d_LX61lINMfSBkpxzLpr-dOsmdUbOf4mp3iy5M7sxnbXhILFuXfd93e6wfKO2h8nw9M2qU8OqKGZmZfJpLUbTlbLrKiYvW0LsvpdBPuoSSa_tfls1O8lb_vI8S_bMukOOujubm7qnniLkVgbfr2ICfXpWTVntMzXVS4TH49FND4XPqnTZrpwGmtuqDPJB263Ux2w0d2VmTXmFH1k-OHohtWHGe7ERsz2IB8I-2LPBFHuOJQf_xUe-Y1Et4bg0zIuItcSprau5BzdYXnl8HTVxC8v_9VGCesSKYG7fncH0_wHsjNqe"
# decode string + decompress zip
depthMapData = parse(s)
# parse first bytes to describe data
header = parseHeader(depthMapData)
# parse bytes into planes of float values
data = parsePlanes(header, depthMapData)
# compute position and values of pixels
depthMap = computeDepthMap(header, data["indices"], data["planes"])
# process float 1D array into int 2D array with 255 values
im = depthMap["depthMap"]
im[np.where(im == max(im))[0]] = 255
if min(im) < 0:
    im[np.where(im < 0)[0]] = 0
im = im.reshape((depthMap["height"], depthMap["width"])).astype(int)
# display image
plt.imshow(im)
plt.show()
