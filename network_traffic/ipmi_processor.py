#! /usr/bin/env python3

# from concurrent.futures import thread
# import enum
# from struct import pack
import sys
import logging
import argparse
import socket
import time
import os
import inspect
import event_list
# from threading import Thread
import pypacker.pypacker as pypacker
from pypacker.pypacker import Packet
from pypacker import ppcap
from pypacker import psocket
from pypacker.layer12 import ethernet
from pypacker.layer3 import ip
from pypacker.layer4 import udp
import hashlib
import hmac
from Crypto.Cipher import AES

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# net function codes from https://openipmi.sourceforge.io/IPMI.pdf chapter 5, not all supported for now
netFun_codes = {
    b"\x00": "Chassis Request",
    b"\x01": "Chassis Response",
    b"\x02": "Bridge Request",
    b"\x03": "Bridge Response",
    b"\x04": "Sensor/Event Request",
    b"\x05": "Sensor/Event Response",
    b"\x06": "Application Request",
    b"\x07": "Application Response",
}


error_codes = {  # every response has a one byte error code, alway the first byte of the message: exhaustive list Table 5.2. in https://openipmi.sourceforge.io/IPMI.pdf
    # or table 5 in https://www.intel.com/content/www/us/en/products/docs/servers/ipmi/ipmi-second-gen-interface-spec-v2-rev1-1.html
    b"\x00": "Command Completed Normally",
    b"\xC0": "Node Busy",
    b"\xC1": "Invalid Command",
}


chasis_control_commands = {  # table 28 of https://www.intel.com/content/www/us/en/products/docs/servers/ipmi/ipmi-second-gen-interface-spec-v2-rev1-1.html
    b"": "Get Chassis Status",
    b"\x00": "power down",
    b"\x01": "power up",
    b"\x02": "power cycle",  # optional
    b"\x03": "hard reset",
    b"\x04": "pulse Diagnostic interrupt",  # optional
    b"\x05": "initiate a soft-shutdown of OS via ACPI by emulating a fatal overtemperature.",  # optional others are reserved
}


def build_ipmi_packet(frame, ts):
    """Builds dict of IPMI packet along with a timestamp, if it is not IPMI it returns None """
    packet = None
    if frame[ethernet.Ethernet] is not None and frame[ip.IP] is not None and frame[udp.UDP] is not None:
        if frame[udp.UDP].dport == 623 or frame[udp.UDP].sport == 623:
            packet = {}
            packet["ts"] = ts
            packet["src_ip"] = frame[ip.IP].src
            packet["dst_ip"] = frame[ip.IP].dst
            packet["src_port"] = frame[udp.UDP].sport
            packet["dst_port"] = frame[udp.UDP].dport
            packet["rmcp_payload"] = parse_rmcp_packet(frame[udp.UDP].body_bytes)
    return packet


def read_pcap(f_path):
    """Processes pcap file, returns list of al impi datagrams and occured timestamps"""
    pcap = ppcap.Reader(f_path)
    packets = []
    for ts, buf in pcap:  # ts = timestamp in nanoseconds
        frame = ethernet.Ethernet(buf)
        packet = build_ipmi_packet(frame, ts)
        if packet is not None:
            packets.append(packet)
    return packets


def parse_rmcp_packet(payload):
    ipmi = {}
    ipmi["version"] = payload[0]
    ipmi["reserved"] = payload[1]
    ipmi["sequence"] = payload[2]
    ipmi["type"] = payload[3]  # Normal RMCP, class IPMI 0x07
    ipmi["auth_type"] = payload[4]  # 0: None,
    # if auth_type = 0: normal ipmi packet
    if ipmi["auth_type"] == 0:
        ipmi["sequence_no"] = payload[5:9]
        ipmi["session_id"] = payload[9:13]
        ipmi["msg_length"] = payload[13]
        ipmi["mgs"] = payload[13:]
    elif ipmi["auth_type"] == 6:
        ipmi["payload_type"] = payload[5]
        ipmi["session_id"] = payload[6:10]
        ipmi["session_sequence_no"] = payload[10:14]
        ipmi["msg_length"] = int.from_bytes(payload[14:15], byteorder="big")
        ipmi["msg"] = payload[16:(15+ipmi["msg_length"]+1)]
        ipmi["trailer"] = payload[(15+ipmi["msg_length"]+1):]
    return ipmi


def handshake_extractor(packets, password):
    """Receives a list of IPMI packets and extracts the handshake along with cryptographic material
    Follows: https://github.com/beingj/hash/blob/master/RMCP%2B%20Packet%20decrypt%20and%20authcode.org"""
    # payload types and contents (*needed)
    # 0x10: RMCP+ Open Session Request
    # * 0x11: RMCP+ Open Session Response, Information: Naming of Confidentiality, Integrity and authentication algorithms
    # * 0x12: RAKP Message 1, Information: Rm, Rolem, Ulengthm, UNamem
    # * 0x13: RAKP Message 2, Information: Rc
    # 0x14: RAKP Message 3
    # 0x15: RAKP Message 4
    # * 0xC0: the following packet should be decrypted: Content: IV for decrypting AES-CBC-128 (confidentiality header in packet)
    # Assuming:
    # According to IPMI SPEC Table 13-8, RMCP/RMCP+ Packet Format for IPMI via Ethernet, packet #9 can be splited to:
    #
    # Auth Type/Format 06 => Format = RMCP+ (IPMI v2.0 only)
    # Payload Type c0 => payload is encrypted, payload is authenticated, payload is IPMI Message (per Table 13-16)
    # IPMI v2.0 RMCP+ Session ID: from packet to be decrypted
    # Session Sequence Number: from packet to be decrypted
    # IPMI Msg/Payload length: from packet to be decrypted
    # Confidentiality Header (I will explain later why it’s 16 bytes): first 16 bytes of message body from packet to be decrypted
    # Payload Data (encrypted. 16 bytes, plus 16 bytes Header, total 0x20 bytes payload) remaining message body bytes after confidentiality header
    # Confidentiality Trailer none => already padded to payload and encrypted
    # Integrity PAD ff ff
    # Pad Length 02
    # Next Header 07 => always = 07h for RMCP+ packets
    # AuthCode (Integrity Data) 1c 73 0c 19 38 51 72 14 34 38 d9 11
    #
    # SIK = HMAC KG (Rm | Rc | RoleM | ULengthM | <UNameM>)
    # About KG, SPEC says: “Note that K[UID] is used in place of Kg if ‘one-key’ logins are being used.” Then What’s K[UID]? According to IPMI SPEC 13.31 RMCP+ Authenticated Key-Exchange Protocol (RAKP):
    #
    # A user needs to know both KG and a user password (key K[UID]) to establish a session, unless the channel is configured with a ‘null’ KG, in which case the user key (K[UID]) is used in place of KG in the algorithms.
    # => KG = password => ASCII, pad to 16 bytes!
    kg = password.ljust(16, b"\00")
    for p in packets:
        if p["payload_type"] == 17:  # 0x11: Open Session Response
            auth_algo = p["msg"][16]  # value 1: RAKP-HMAC-SHA1 (table 13-17)
            int_algo = p["msg"][24]  # value 1: HMAC-SHA1-96 (table 13-18)
            conf_algo = p["msg"][32]  # value 1: AES-CBC-128 (table 13-19)
        if p["payload_type"] == 18:  # 0x12
            rm = p["msg"][8:24]
            rolem = p["msg"][24]
            ulengthm = p["msg"][27]
            unamem = p["msg"][28:(28 + ulengthm)]
        if p["payload_type"] == 19:  # 0x13
            rc = p["msg"][8:24]
    const2 = b"".ljust(20, b'\x02')
    # K2 = HMAC SIK (const 2)
    d = b"".join([rm, rc, bytes([rolem]), bytes([ulengthm]), unamem])
    sik = hmac.new(kg, d, hashlib.sha1).digest()
    k2 = hmac.new(sik, const2, hashlib.sha1).digest()[:16]
    return sik, k2


def parse_ipmi(payload, k, request):
    message = {}
    iv = payload["msg"][:16]
    ipmi_cypher = payload["msg"][16:]
    auth_code = payload["trailer"][4:]
    cypher = AES.new(k, AES.MODE_CBC, iv)
    plaintext = cypher.decrypt(ipmi_cypher)
    padding = plaintext[-1]
    m = plaintext[:len(plaintext)-padding-1]
    if request:
        message["rsSA"] = m[0]
        # TODO Extract first 6 bit for functioncode, last 2 bits for rsLUN
        message["netFN/rsLUN"] = m[1]
        LUN_mask = 0b00000011
        netFN_mask = 0b11111100
        message["rsLUN"] = LUN_mask & message["netFN/rsLUN"]
        message["netFN"] = bytes([(netFN_mask & message["netFN/rsLUN"]) >> 2])
        message["checksum1"] = m[2]
        message["rqSA"] = m[3]
        # TODO extract first 6 bits for rqSeq and last 2 bits for rqLUN
        message["rqSeq/rqLUN"] = m[4]
        message["rqLUN"] = LUN_mask & message["rqSeq/rqLUN"]
        message["rqSeq"] = netFN_mask & message["rqSeq/rqLUN"] >> 2
        message["command"] = bytes([m[5]])
        # Attention: little endian here in the bytes
        message["command_bytes"] = m[6:-1]
        message["checksum2"] = m[-1]
    else:
        message["rqSA"] = m[0]
        # TODO Extract first 6 bit for functioncode, last 2 bits for rqLUN
        message["netFN/rqLUN"] = m[1]
        LUNmask = 0b00000011
        netFN_mask = 0b11111100
        message["rqLUN"] = LUNmask & message["netFN/rqLUN"]
        message["netFN"] = bytes([(netFN_mask & message["netFN/rqLUN"]) >> 2])
        message["checksum1"] = m[2]
        message["rsSA"] = m[3]
        # TODO extract first 6 bits for rsSeq and last 2 bits for rsLUN
        message["rqSeq/rsLUN"] = m[4]
        message["rsLUN"] = LUNmask & message["rqSeq/rsLUN"]
        message["rqSeq"] = netFN_mask & message["rqSeq/rsLUN"] >> 2
        message["command"] = bytes([m[5]])
        # Attention: little endian here in the bytes
        message["response_data"] = m[6:-1]
        message["checksum2"] = m[-1]
    return message


def process_packet(p, partners, password):
    out = "{} \t {}:{} -> {}:{}".format(p["ts"], ".".join(str(i) for i in p["src_ip"]), p["src_port"], ".".join(str(i) for i in p["dst_ip"]), p["dst_port"])
    parties = frozenset([p["src_ip"], p["dst_ip"]])
    inf = {}
    if "rmcp_payload" in p and "payload_type" in p["rmcp_payload"]:
        if parties not in partners:
            partners[parties] = {}
            partners[parties]["hs"] = []
            partners[parties]["k2"] = None
            partners[parties]["sik"] = None
        if p["rmcp_payload"]["payload_type"] == 16:
            partners[parties]["hs"].append(p["rmcp_payload"]) if p["rmcp_payload"] not in partners[parties]["hs"] else None
            out += " RMCP+ Open Session Request"
        if p["rmcp_payload"]["payload_type"] == 17:
            partners[parties]["hs"].append(p["rmcp_payload"]) if p["rmcp_payload"] not in partners[parties]["hs"] else None
            out += " RMCP+ Open Session Response"
        if p["rmcp_payload"]["payload_type"] == 18:
            partners[parties]["hs"].append(p["rmcp_payload"]) if p["rmcp_payload"] not in partners[parties]["hs"] else None
            out += " RAKP Message 1"
        if p["rmcp_payload"]["payload_type"] == 19:
            partners[parties]["hs"].append(p["rmcp_payload"]) if p["rmcp_payload"] not in partners[parties]["hs"] else None
            out += " RAKP Message 2"
        if p["rmcp_payload"]["payload_type"] == 20:
            partners[parties]["hs"].append(p["rmcp_payload"]) if p["rmcp_payload"] not in partners[parties]["hs"] else None
            out += " RAKP Message 3"
        if p["rmcp_payload"]["payload_type"] == 21:
            partners[parties]["hs"].append(p["rmcp_payload"]) if p["rmcp_payload"] not in partners[parties]["hs"] else None
            out += " RAKP Message 4"
        if p["rmcp_payload"]["payload_type"] == 192:
            if partners[parties]["k2"] is not None:
                try:
                    payload = parse_ipmi(p["rmcp_payload"], partners[parties]["k2"], p["dst_port"] == 623)
                    inf["ts"] = p["ts"]
                    if payload["netFN"] == b"\x00" or payload["netFN"] == b"\x01":  # chassis commands
                        inf["netFN"] = payload["netFN"]
                        if p["dst_port"] == 623:
                            inf["device"] = ".".join(str(i) for i in p["dst_ip"])
                            inf["req"] = True
                            out += " IPMI Request: {}: ".format(netFun_codes[payload["netFN"]])
                            if payload["netFN"] == b"\x00":
                                inf["netFN"] = netFun_codes[payload["netFN"]].replace(" ", "-")  # string mapping
                                inf["command_bytes"] = payload["command_bytes"]
                                inf["command_bytes"] = chasis_control_commands[payload["command_bytes"]].replace(" ", "-")  # string mapping
                                out += chasis_control_commands[payload["command_bytes"]]
                        else:
                            inf["device"] = ".".join(str(i) for i in p["src_ip"])
                            inf["req"] = False
                            inf["response_code"] = bytes([payload["response_data"][0]])
                            out += " IPMI Response: {}: ".format(netFun_codes[payload["netFN"]])
                            out += error_codes[bytes([payload["response_data"][0]])]  # first byte (endianness last) has response code
                            if len(payload["response_data"]) > 1:
                                out += " +more unparsed response data"
                                inf["response_bytes"] = payload["response_data"]
                        print(out)
                except BaseException as error:
                    print("Error decrypting packet")
                    print("Output when error occured", out)
                    print("Error: ", error)
        if parties in partners:
            if len(partners[parties]["hs"]) == 6:
                print("Handshake captured, decrypting IPMI payload with generated key material")
                partners[parties]["sik"], partners[parties]["k2"] = handshake_extractor(partners[parties]["hs"], password)
                partners[parties]["hs"] = []
    if inf == {} or len(inf.keys()) == 1:
        inf = None
    return (inf, partners)


def process_captured_traffic(pcap_file, password):
    partners = {}
    pcap = read_pcap(pcap_file)
    events = []
    for p in pcap:
        (msg, partners) = process_packet(p, partners, password)
        if msg is not None:
            events.append(msg)
    return events


def process_live_traffic(iface, password):
    partners = {}
    try:
        psock = psocket.SocketHndl(iface_name=iface, timeout=None)
        print("listening in on interface {}".format(iface))
        for raw_bytes in psock:
            packet = build_ipmi_packet(ethernet.Ethernet(raw_bytes), time.time_ns())
            if packet is not None:
                (msg, partners) = process_packet(packet, partners, password)
                if msg is not None:
                    msg["record_until"] = msg["ts"]+3000000
                    event_list.events.append(msg)
        psock.close()
    except socket.error as e:
        print("Root privileges required for live traffic analysis, please restart the script with root privileges")


def parse_arguments():
    parser = argparse.ArgumentParser(description="IPMI traffic processor and decryptor")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-f", "--file", help="Path to file which is supposed to be processed")
    mode.add_argument("-i", "--interface", help="name of interface on which the precessor should listen to IPMI traffic")
    parser.add_argument("-p", "--password", help="Specify password, if parameter not used \"ADMIN\" is used as password", required=False)
    args = vars(parser.parse_args())
    if args["password"] is None:
        args["password"] = b"ADMIN"  # the default IPMI password is ADMIN
    else:
        args["password"] = bytes(args["password"], "utf-8")
    return args


events = []


def main(args):
    args = parse_arguments()
    if args["file"] is not None:
        process_captured_traffic(args["file"], args["password"])
    else:
        process_live_traffic(args["interface"], args["password"])


if __name__ == "__main__":
    main(sys.argv)
