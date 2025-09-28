#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int32_t std_write(int32_t fd, void* data, int32_t len) {
    return write(fd, data, len);
}

int32_t std_read(int32_t fd, void* data, int32_t len) {
    printf("read: fd %d, len: %d\n", fd, len);
    return read(fd, data, len);
}

char* std_getenv(void* name) { return getenv(name); }

char* std_strerror() { return strerror(errno); }

void print_strerror() { printf("%s\n", std_strerror()); }

int32_t std_tcp_listen(char* addr, int32_t port) {
    int32_t socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == -1) {
        print_strerror();
        return -1;
    }

    struct sockaddr_in sockaddr = {
        AF_INET,
        htons(port),
        {inet_addr(addr)},
    };

    if (bind(socket_fd, (struct sockaddr*)&sockaddr, sizeof(sockaddr)) == -1) {
        print_strerror();
        return -1;
    }

    if (listen(socket_fd, 128) == -1) {
        print_strerror();
        return -1;
    }

    return socket_fd;
}

typedef struct {
    uint32_t addr;
    int32_t fd;
    uint16_t port;
} TcpConn;

TcpConn TcpConn_err() {
    TcpConn v = {
        .fd = -1,
        .addr = 0,
        .port = 0,
    };
    return v;
}

TcpConn std_tcp_accept(int32_t fd) {
    printf("tcp accept\n");
    struct sockaddr_in client_addr;
    socklen_t client_addr_size = sizeof(client_addr);

    int32_t client_fd;
    if ((client_fd = accept(fd, (struct sockaddr*)&client_addr,
                            &client_addr_size)) == -1) {
        printf("tcp error: %s\n", std_strerror());
        return TcpConn_err();
    }

    TcpConn tcp_conn = {
        .addr = ntohl(client_addr.sin_addr.s_addr),
        .fd = client_fd,
        .port = ntohs(client_addr.sin_port),
    };

    printf("addr: %d, fd: %d, port: %d\n", tcp_conn.addr, tcp_conn.fd,
           tcp_conn.port);

    return tcp_conn;
}
